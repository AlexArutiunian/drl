#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Экстрактор фич из .npy c расширенными динамическими, частотными, L/R-фазовыми,
угловыми и ROI-пулинг признаками для мультиклассификации по зонам ног.

Особенности:
  - (T, N, 3) или (T, 3*N) вход; NaN/Inf -> 0; самая длинная ось трактуется как время.
  - Схема суставов: --schema schema_joints.json (список из N имён).
  - Фильтрация по CSV со списком файлов (--filelist_csv, колонка --filelist_col).
  - Базовые статистики по x,y,z, mag, vmag, amag + ZCR и RMS.
  - Новые признаки:
      * FFT-спектральные (центроид/ширина/rolloff/пиковая частота/мощность, полосы 0–1/1–2/2–4 Гц)
      * Пиковые (число пиков/сек, межпиковое время, размах)
      * L/R: diff, symmetry index, корреляция, лаг максимальной кросскорреляции по vmag
      * «jerk» (третья разность для mag)
      * Углы колена L/R + их спектральные, и асимметрия углов
      * ROI-пулинг (ankle_foot, knee, thigh, shin_calf, pelvis)
  - n_frames не сохраняем; оставляем n_joints, lr_pairs_count.

Пример:
  python feats.py \
    --data_dir /kaggle/working/npy_split_aug/train_chunks_aug \
    --out_csv  /kaggle/working/features.csv \
    --schema   detect_inj/schema_joints.json \
    --filelist_csv /kaggle/working/train_meta.csv \
    --filelist_col new_filename \
    --fps 30
"""

import os, json, argparse, re
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------- name helpers ----------
def _sanitize_names(names: List[str]) -> List[str]:
    out = []
    for i, s in enumerate(names):
        s = str(s).strip().replace(" ", "_").replace("/", "_")
        out.append(s if s else f"j{i}")
    return out

def load_schema(schema_path: Optional[str], n_joints_detected: int) -> List[str]:
    if schema_path is None:
        return [f"j{i}" for i in range(n_joints_detected)]
    names = json.load(open(schema_path, "r", encoding="utf-8"))
    if not isinstance(names, list) or len(names) != n_joints_detected:
        raise ValueError(f"Schema length ({len(names)}) != detected joints ({n_joints_detected})")
    return _sanitize_names(names)


# ---------- loading ----------
def load_npy_to_TxNx3(path: str, n_from_schema: Optional[int]) -> np.ndarray:
    A = np.load(path, allow_pickle=False)
    # Выровнять во времени и координатах
    if A.ndim == 3 and A.shape[2] == 3:
        pass
    elif A.ndim == 2 and A.shape[1] % 3 == 0:
        N = n_from_schema if n_from_schema is not None else (A.shape[1] // 3)
        if 3 * N != A.shape[1]:
            raise ValueError(f"Width {A.shape[1]} не делится на 3*N (N={N})")
        A = A.reshape(A.shape[0], N, 3)
    else:
        raise ValueError(f"Unexpected array shape {A.shape}. Ожидалось (T,N,3) или (T,3*N).")

    # Самую длинную ось считаем временем
    if np.argmax(A.shape) != 0:
        A = np.moveaxis(A, np.argmax(A.shape), 0)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return A

def select_frames(A: np.ndarray, max_len: int, mode: str = "uniform", len_slack: int = 0) -> np.ndarray:
    if max_len is None or max_len <= 0:
        return A
    T = A.shape[0]
    if T <= max_len + max(0, len_slack):
        return A
    if mode == "head":   return A[:max_len]
    if mode == "tail":   return A[-max_len:]
    if mode == "center":
        start = max(0, (T - max_len) // 2); return A[start:start + max_len]
    if mode == "random":
        start = np.random.randint(0, T - max_len + 1); return A[start:start + max_len]
    idx = np.linspace(0, T - 1, num=max_len, dtype=int)
    return A[idx]


# ---------- basic stats ----------
def _basic_stats(x: np.ndarray) -> dict:
    x = x.astype(np.float32, copy=False)
    n = x.size
    m  = float(np.mean(x)); sd = float(np.std(x))
    mn = float(np.min(x));  mx = float(np.max(x)); pt = float(np.ptp(x))
    med = float(np.median(x))
    q25 = float(np.percentile(x, 25)); q75 = float(np.percentile(x, 75))
    iqr = float(q75 - q25); mad = float(np.median(np.abs(x - med)))
    rms = float(np.sqrt(np.mean(x * x)))
    if sd > 0:
        z = (x - m) / sd
        sk = float(np.mean(z ** 3)); ku = float(np.mean(z ** 4) - 3.0)
    else:
        sk, ku = 0.0, 0.0
    if n > 1:
        c = np.corrcoef(x[:-1], x[1:])
        ac1 = float(c[0, 1]) if np.isfinite(c).all() else 0.0
    else:
        ac1 = 0.0
    cov = float(sd / (abs(m) + 1e-8))
    iqr_over_range = float(iqr / (pt + 1e-8))
    return {
        "mean": m, "std": sd, "min": mn, "max": mx, "ptp": pt,
        "median": med, "q25": q25, "q75": q75, "iqr": iqr,
        "mad": mad, "skew": sk, "kurt": ku, "rms": rms, "ac1": ac1,
        "cov": cov, "iqr_over_range": iqr_over_range
    }

def _zcr(x: np.ndarray) -> float:
    if x.size < 2: return 0.0
    s = np.sign(x); s[s == 0] = 1
    return float(np.mean(s[1:] != s[:-1]))

def _diff(x: np.ndarray, order: int = 1) -> np.ndarray:
    if x.size <= order: return np.zeros_like(x[:1])
    return np.diff(x, n=order)

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2: return 0.0
    sd_a, sd_b = np.std(a), np.std(b)
    if sd_a == 0 or sd_b == 0: return 0.0
    c = np.corrcoef(a, b)
    return float(c[0, 1]) if np.isfinite(c).all() else 0.0

def _add_stats(out: dict, prefix: str, x: np.ndarray):
    st = _basic_stats(x)
    for k, v in st.items():
        out[f"{prefix}_{k}"] = float(v)


# ---------- spectral / peaks / xcorr ----------
def _fft_feats(x: np.ndarray, fps: float) -> dict:
    x = np.asarray(x, np.float32)
    x = x - x.mean()
    T = x.size
    if T < 16 or fps <= 1e-6:
        return {"spec_centroid":0.0,"spec_bw":0.0,"spec_rolloff85":0.0,
                "spec_peak_freq":0.0,"spec_peak_pow":0.0,
                "band_0_1":0.0,"band_1_2":0.0,"band_2_4":0.0}
    w = np.hanning(T).astype(np.float32)
    X = np.fft.rfft(x * w)
    P = (np.abs(X) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0/fps).astype(np.float32)

    Ps = P / (P.sum() + 1e-8)
    spec_centroid = float((freqs * Ps).sum())
    spec_bw = float(np.sqrt(((freqs - spec_centroid) ** 2 * Ps).sum()))

    cumsum = np.cumsum(Ps)
    idx_roll = int(np.searchsorted(cumsum, 0.85))
    spec_rolloff85 = float(freqs[min(idx_roll, len(freqs)-1)])

    i_peak = int(P.argmax())
    spec_peak_freq = float(freqs[i_peak])
    spec_peak_pow  = float(Ps[i_peak])

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(Ps[m].sum())
    return {
        "spec_centroid": spec_centroid,
        "spec_bw": spec_bw,
        "spec_rolloff85": spec_rolloff85,
        "spec_peak_freq": spec_peak_freq,
        "spec_peak_pow": spec_peak_pow,
        "band_0_1": band(0.0, 1.0),
        "band_1_2": band(1.0, 2.0),
        "band_2_4": band(2.0, 4.0),
    }

def _peak_feats(x: np.ndarray, fps: float) -> dict:
    x = np.asarray(x, np.float32)
    T = x.size
    if T < 5 or fps <= 1e-6:
        return {"peaks_per_s":0.0,"mean_interpeak_s":0.0,"p2p":float(np.ptp(x))}
    dx = np.diff(x)
    peaks = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0] + 1
    n = len(peaks)
    peaks_per_s = float(n / (T / fps))
    if n >= 2:
        inter = np.diff(peaks) / fps
        mean_ip = float(inter.mean())
    else:
        mean_ip = float(T / fps)
    return {"peaks_per_s": peaks_per_s, "mean_interpeak_s": mean_ip, "p2p": float(np.ptp(x))}

def _xcorr_max_lag(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, np.float32) - np.mean(a)
    b = np.asarray(b, np.float32) - np.mean(b)
    sa, sb = a.std()+1e-8, b.std()+1e-8
    a /= sa; b /= sb
    T = min(a.size, b.size)
    a, b = a[:T], b[:T]
    if T < 8:
        return 0.0, 0.0
    c = np.correlate(a, b, mode="full")  # длина 2T-1
    k = int(c.argmax())
    lag = k - (T - 1)
    lag_norm = float(lag / T)                 # [-1..1]
    corr_max = float(c.max() / (T))           # ~[0..1]
    return lag_norm, corr_max


# ---------- L/R pairing ----------
_L_PAT = re.compile(r'(^|[_\-\s\(\[])l(eft)?([_\-\s\)\]]|$)', re.IGNORECASE)
_R_PAT = re.compile(r'(^|[_\-\s\(\[])r(ight)?([_\-\s\)\]]|$)', re.IGNORECASE)

def _split_side(name: str) -> Tuple[Optional[str], str]:
    if _L_PAT.search(name):
        base = _L_PAT.sub(r'\1\3', name)
        return "L", re.sub(r'__+', '_', base).strip('_')
    if _R_PAT.search(name):
        base = _R_PAT.sub(r'\1\3', name)
        return "R", re.sub(r'__+', '_', base).strip('_')
    return None, name

def detect_lr_pairs(joint_names: List[str]) -> List[Tuple[int,int,str]]:
    buckets: Dict[str, Dict[str,int]] = {}
    for idx, nm in enumerate(joint_names):
        side, base = _split_side(nm)
        base_norm = base.lower()
        if base_norm not in buckets:
            buckets[base_norm] = {}
        if side is not None:
            buckets[base_norm][side] = idx
    pairs = []
    for base_norm, d in buckets.items():
        if "L" in d and "R" in d:
            pairs.append((d["L"], d["R"], base_norm))
    return pairs

def _pair_feats_dict(prefix: str, L: np.ndarray, R: np.ndarray) -> dict:
    out = {}
    eps = 1e-8
    diff = L - R
    _add_stats(out, f"{prefix}_diff", diff)
    si = 2.0 * diff / (L + R + eps)   # symmetry index
    _add_stats(out, f"{prefix}_si", si)
    out[f"{prefix}_corr"] = _safe_corr(L, R)
    return out


# ---------- angles ----------
def _angle_at(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """угол в точке p2 (T,3) между (p1-p2) и (p3-p2) в радианах"""
    v1 = p1 - p2; v2 = p3 - p2
    a = np.sum(v1*v2, axis=1)
    b = np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1) + 1e-8
    c = np.clip(a/b, -1.0, 1.0)
    return np.arccos(c).astype(np.float32)

def _find_idxs(joint_names: List[str], keywords: List[str]) -> Optional[int]:
    kw = [k.lower() for k in keywords]
    for i, nm in enumerate(joint_names):
        s = nm.lower()
        if all(k in s for k in kw):
            return i
    return None

def _try_knee_angles(A: np.ndarray, names: List[str], side: str) -> Optional[np.ndarray]:
    tag = "l" if side.upper() == "L" else "r"
    i_hip   = _find_idxs(names, [tag, "hip"]) or _find_idxs(names, [tag, "pelvis"])
    i_knee  = _find_idxs(names, [tag, "knee"])
    i_ankle = _find_idxs(names, [tag, "ankle"]) or _find_idxs(names, [tag, "foot"])
    if None in (i_hip, i_knee, i_ankle):
        return None
    H = A[:, i_hip, :]; K = A[:, i_knee, :]; A2 = A[:, i_ankle, :]
    return _angle_at(H, K, A2)  # коленный угол


# ---------- ROI pooling ----------
def _roi_indices(names: List[str]) -> Dict[str, List[int]]:
    groups = {"ankle_foot":[], "knee":[], "thigh":[], "shin_calf":[], "pelvis":[]}
    for i, nm in enumerate(names):
        s = nm.lower()
        if "ankle" in s or "foot" in s: groups["ankle_foot"].append(i)
        if "knee"  in s: groups["knee"].append(i)
        if "thigh" in s: groups["thigh"].append(i)
        if "shin"  in s or "calf" in s: groups["shin_calf"].append(i)
        if "pelvis" in s or "hip" in s: groups["pelvis"].append(i)
    return groups

def _roi_pool_feats(A: np.ndarray, idxs: List[int], prefix: str, fps_current: float) -> dict:
    if not idxs: return {}
    X = A[:, idxs, :]                    # (T, M, 3)
    mag = np.linalg.norm(X, axis=2)      # (T, M)
    vmag = np.diff(mag, axis=0)          # скорость по магн. (T-1, M)
    out = {}
    _add_stats(out, f"{prefix}_mag_mean_over_j", mag.mean(axis=1))
    _add_stats(out, f"{prefix}_vmag_mean_over_j", vmag.mean(axis=1))
    roi_sig = mag.mean(axis=1)
    out.update({f"{prefix}_{k}": v for k, v in _fft_feats(roi_sig, fps_current).items()})
    return out


# ---------- read allowed filenames from CSV ----------
def read_allowed_filenames(csv_path: Path, column: str, encoding: str) -> Tuple[Set[str], Set[str], Set[str]]:
    df = pd.read_csv(csv_path, encoding=encoding)
    if column not in df.columns:
        match = next((c for c in df.columns if c.strip().lower() == column.strip().lower()), None)
        if match is None:
            raise SystemExit(f"[err] В {csv_path} нет колонки '{column}'. Найдено: {list(df.columns)}")
        column = match
    vals = (
        df[column]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .replace({"": None})
        .dropna()
        .tolist()
    )
    basenames = set(os.path.basename(v).lower() for v in vals)
    stems = set(Path(v).stem.lower() for v in vals)
    raw_paths = set(v.lower() for v in vals)
    return basenames, stems, raw_paths

def filter_files_by_list(root: Path, basenames: Set[str], stems: Set[str], raw_paths: Set[str]) -> Tuple[List[Path], int]:
    all_files = list(root.rglob("*.npy")) + list(root.rglob("*.NPY"))
    if not all_files:
        return [], 0
    basenames_l = set(basenames); stems_l = set(stems); raw_paths_l = set(raw_paths)
    matched: List[Path] = []
    for p in all_files:
        name_l = p.name.lower(); stem_l = p.stem.lower(); path_l = str(p.resolve()).lower()
        if (name_l in basenames_l) or (stem_l in stems_l) or (path_l in raw_paths_l):
            matched.append(p)
    n_requested = len(basenames_l | stems_l | raw_paths_l)
    return sorted(matched), n_requested


# ---------- feature extraction ----------
def extract_features(A: np.ndarray, joint_names: List[str], fps_current: float) -> dict:
    T, N, _ = A.shape
    out = {"n_joints": int(N)}  # n_frames не записываем

    # --- per-joint
    for j in range(N):
        name = joint_names[j]
        x, y, z = A[:, j, 0], A[:, j, 1], A[:, j, 2]
        _add_stats(out, f"{name}_x", x)
        _add_stats(out, f"{name}_y", y)
        _add_stats(out, f"{name}_z", z)
        out[f"{name}_xy_corr"] = _safe_corr(x, y)
        out[f"{name}_yz_corr"] = _safe_corr(y, z)
        out[f"{name}_xz_corr"] = _safe_corr(x, z)

        mag = np.sqrt(x*x + y*y + z*z)
        _add_stats(out, f"{name}_mag", mag)

        vx, vy, vz = _diff(x, 1), _diff(y, 1), _diff(z, 1)
        vmag = _diff(mag, 1)
        _add_stats(out, f"{name}_vx", vx); _add_stats(out, f"{name}_vy", vy); _add_stats(out, f"{name}_vz", vz); _add_stats(out, f"{name}_vmag", vmag)
        out[f"{name}_vx_zcr"] = _zcr(vx); out[f"{name}_vy_zcr"] = _zcr(vy); out[f"{name}_vz_zcr"] = _zcr(vz); out[f"{name}_vmag_zcr"] = _zcr(vmag)

        ax, ay, az = _diff(x, 2), _diff(y, 2), _diff(z, 2)
        amag = _diff(mag, 2)
        _add_stats(out, f"{name}_ax", ax); _add_stats(out, f"{name}_ay", ay); _add_stats(out, f"{name}_az", az); _add_stats(out, f"{name}_amag", amag)

        # jerk (третья разность) по magnitudes
        jerk = _diff(mag, 3)
        _add_stats(out, f"{name}_jerk", jerk)

        # частотные и пиковые признаки
        out.update({f"{name}_mag_{k}": v for k, v in _fft_feats(mag, fps_current).items()})
        out.update({f"{name}_vmag_{k}": v for k, v in _fft_feats(vmag, fps_current).items()})
        out.update({f"{name}_magpk_{k}": v for k, v in _peak_feats(mag, fps_current).items()})

    # --- L/R асимметрия
    pairs = detect_lr_pairs(joint_names)
    out["lr_pairs_count"] = int(len(pairs))

    for iL, iR, base in pairs:
        Lx, Ly, Lz = A[:, iL, 0], A[:, iL, 1], A[:, iL, 2]
        Rx, Ry, Rz = A[:, iR, 0], A[:, iR, 1], A[:, iR, 2]
        Lmag = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz); Rmag = np.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
        Lvmag, Rvmag = _diff(Lmag,1), _diff(Rmag,1)

        base_tag = f"PAIR_{base}"
        out.update(_pair_feats_dict(f"{base_tag}_x", Lx, Rx))
        out.update(_pair_feats_dict(f"{base_tag}_y", Ly, Ry))
        out.update(_pair_feats_dict(f"{base_tag}_z", Lz, Rz))
        out.update(_pair_feats_dict(f"{base_tag}_mag", Lmag, Rmag))

        # фазовый лаг по vmag
        lag, cc = _xcorr_max_lag(Lvmag, Rvmag)
        out[f"{base_tag}_vmag_xlag"] = lag
        out[f"{base_tag}_vmag_xcorrmax"] = cc

    # --- ROI pooling
    rois = _roi_indices(joint_names)
    for k, idxs in rois.items():
        out.update(_roi_pool_feats(A, idxs, f"ROI_{k}", fps_current))

    # --- коленные углы (если доступны)
    aL = _try_knee_angles(A, joint_names, "L")
    aR = _try_knee_angles(A, joint_names, "R")
    if aL is not None:
        _add_stats(out, "knee_L_angle", aL)
        out.update({f"knee_L_angle_{k}": v for k, v in _fft_feats(aL, fps_current).items()})
    if aR is not None:
        _add_stats(out, "knee_R_angle", aR)
        out.update({f"knee_R_angle_{k}": v for k, v in _fft_feats(aR, fps_current).items()})
    if (aL is not None) and (aR is not None):
        _add_stats(out, "knee_angle_diff", aL - aR)

    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Экстрактор фич (статистика+динамика+частоты+L/R+углы+ROI) с фильтрацией по списку файлов")
    ap.add_argument("--data_dir", required=True, help="Папка с .npy (рекурсивно)")
    ap.add_argument("--out_csv",  required=True, help="Путь к итоговому CSV")
    ap.add_argument("--schema",   default=None,  help="schema_joints.json (опционально)")
    ap.add_argument("--filelist_csv", required=True, help="CSV со списком файлов для извлечения")
    ap.add_argument("--filelist_col", default="filename", help="Имя колонки со списком файлов (по умолчанию 'filename')")
    ap.add_argument("--max_len", type=int, default=0, help="макс. число кадров; 0 = без ограничения")
    ap.add_argument("--frame_pick", choices=["uniform","head","tail","center","random"], default="uniform", help="стратегия отбора кадров при обрезке")
    ap.add_argument("--len_slack", type=int, default=500, help="Сжимать только если T > max_len + len_slack")
    ap.add_argument("--encoding", default="utf-8-sig", help="Кодировка CSV файлов")
    ap.add_argument("--fps", type=float, default=30.0, help="Кадровая частота для частотных/ритмических фич (Гц)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    schema_path = args.schema

    basenames, stems, raw_paths = read_allowed_filenames(Path(args.filelist_csv), args.filelist_col, args.encoding)
    files, n_requested = filter_files_by_list(data_dir, basenames, stems, raw_paths)
    if not files:
        raise SystemExit(f"[err] Не найдено ни одного .npy из списка в {data_dir}.\nПроверьте пути/расширения и колонку '{args.filelist_col}'.")
    print(f"[info] Всего в списке (уник.) ~ {n_requested}, найдено под {data_dir}: {len(files)} файлов.")

    rows = []
    for p in tqdm(files, desc="Extract"):
        try:
            # определим N для схемы имён
            A_raw = np.load(p, allow_pickle=False)
            if A_raw.ndim == 3 and A_raw.shape[2] == 3:
                N_detect = A_raw.shape[1]
            elif A_raw.ndim == 2 and A_raw.shape[1] % 3 == 0:
                N_detect = A_raw.shape[1] // 3
            else:
                raise ValueError(f"Unexpected shape {A_raw.shape}")

            names = load_schema(schema_path, N_detect)
            A = load_npy_to_TxNx3(str(p), len(names))
            A = select_frames(A, args.max_len, args.frame_pick, args.len_slack)

            feats = extract_features(A, names, fps_current=args.fps)
            feats.update({"file": str(p.resolve()), "basename": p.name, "stem": p.stem})
            rows.append(feats)
        except Exception as e:
            print(f"[skip] {p}: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("[err] Не удалось извлечь ни одной строки фичей.")

    df = pd.DataFrame(rows)
    # первые колонки — удобные
    first = [c for c in ["file", "basename", "stem", "n_joints", "lr_pairs_count"] if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[done] Saved: {out_csv}  |  rows={len(df)}  cols={len(df.columns)}")


if __name__ == "__main__":
    main()
