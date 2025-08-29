#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Экстрактор фич из .npy c расширенными динамическими и асимметрийными признаками для задачи "левая vs правая".
Изменения:
  - Фильтрация по списку файлов из CSV (filename).
  - Добавлены L/R-парные фичи (diff, symmetry index, corr) для осей x,y,z,mag и их v/a.
  - Удалён признак n_frames (оставлен n_joints).
"""

import os, json, argparse, re
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- helpers ----------
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

def load_npy_to_TxNx3(path: str, n_from_schema: Optional[int]) -> np.ndarray:
    A = np.load(path, allow_pickle=False)
    if A.ndim == 3 and A.shape[2] == 3:
        pass
    elif A.ndim == 2 and A.shape[1] % 3 == 0:
        N = n_from_schema if n_from_schema is not None else (A.shape[1] // 3)
        if 3 * N != A.shape[1]:
            raise ValueError(f"Width {A.shape[1]} не делится на 3*N (N={N})")
        A = A.reshape(A.shape[0], N, 3)
    else:
        raise ValueError(f"Unexpected array shape {A.shape}. Ожидалось (T,N,3) или (T,3*N).")
    return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

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
    return np.diff(x, n=order)

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2: return 0.0
    sd_a, sd_b = np.std(a), np.std(b)
    if sd_a == 0 or sd_b == 0: return 0.0
    c = np.corrcoef(a, b); 
    return float(c[0, 1]) if np.isfinite(c).all() else 0.0

def _add_stats(out: dict, prefix: str, x: np.ndarray):
    st = _basic_stats(x)
    for k, v in st.items():
        out[f"{prefix}_{k}"] = float(v)

# ---------- L/R pairing ----------
_L_PAT = re.compile(r'(^|[_\-\s\(\[])l(eft)?([_\-\s\)\]]|$)', re.IGNORECASE)
_R_PAT = re.compile(r'(^|[_\-\s\(\[])r(ight)?([_\-\s\)\]]|$)', re.IGNORECASE)

def _split_side(name: str) -> Tuple[Optional[str], str]:
    """Возвращает (side, base_name), где side ∈ {'L','R',None}, base_name — имя без маркера стороны."""
    s = name
    if _L_PAT.search(s):
        base = _L_PAT.sub(r'\1\3', s)
        return "L", re.sub(r'__+', '_', base).strip('_')
    if _R_PAT.search(s):
        base = _R_PAT.sub(r'\1\3', s)
        return "R", re.sub(r'__+', '_', base).strip('_')
    return None, s

def detect_lr_pairs(joint_names: List[str]) -> List[Tuple[int,int,str]]:
    """Находит пары (idxL, idxR, base_name) по именам суставов."""
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
    """Фичи для пары сигналов L/R: diff-статистики, symmetry index, corr."""
    out = {}
    eps = 1e-8
    diff = L - R
    _add_stats(out, f"{prefix}_diff", diff)
    si = 2.0 * diff / (L + R + eps)   # symmetry index
    _add_stats(out, f"{prefix}_si", si)
    out[f"{prefix}_corr"] = _safe_corr(L, R)
    return out

def extract_features(A: np.ndarray, joint_names: List[str]) -> dict:
    T, N, _ = A.shape
    out = {"n_joints": int(N)}  # n_frames удалён по запросу

    # --- per-joint базовые фичи (оставляем как было)
    for j in range(N):
        name = joint_names[j]
        x, y, z = A[:, j, 0], A[:, j, 1], A[:, j, 2]
        _add_stats(out, f"{name}_x", x)
        _add_stats(out, f"{name}_y", y)
        _add_stats(out, f"{name}_z", z)
        out[f"{name}_xy_corr"] = _safe_corr(x, y)
        out[f"{name}_yz_corr"] = _safe_corr(y, z)
        out[f"{name}_xz_corr"] = _safe_corr(x, z)
        mag = np.sqrt(x*x + y*y + z*z); _add_stats(out, f"{name}_mag", mag)

        vx, vy, vz = _diff(x, 1), _diff(y, 1), _diff(z, 1); vmag = _diff(mag, 1)
        _add_stats(out, f"{name}_vx", vx); _add_stats(out, f"{name}_vy", vy); _add_stats(out, f"{name}_vz", vz); _add_stats(out, f"{name}_vmag", vmag)
        out[f"{name}_vx_zcr"] = _zcr(vx); out[f"{name}_vy_zcr"] = _zcr(vy); out[f"{name}_vz_zcr"] = _zcr(vz); out[f"{name}_vmag_zcr"] = _zcr(vmag)

        ax, ay, az = _diff(x, 2), _diff(y, 2), _diff(z, 2); amag = _diff(mag, 2)
        _add_stats(out, f"{name}_ax", ax); _add_stats(out, f"{name}_ay", ay); _add_stats(out, f"{name}_az", az); _add_stats(out, f"{name}_amag", amag)

    # --- L/R асимметрия
    pairs = detect_lr_pairs(joint_names)
    out["lr_pairs_count"] = int(len(pairs))

    # аккумулируем глобальные суммы по парам для «общей» асимметрии
    glob_abs_diff_rms = {"x":[], "y":[], "z":[], "mag":[], "vmag":[], "amag":[]}
    for iL, iR, base in pairs:
        # координаты и производные
        Lx, Ly, Lz = A[:, iL, 0], A[:, iL, 1], A[:, iL, 2]
        Rx, Ry, Rz = A[:, iR, 0], A[:, iR, 1], A[:, iR, 2]
        Lmag = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz); Rmag = np.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)

        Lvx, Lvy, Lvz = _diff(Lx,1), _diff(Ly,1), _diff(Lz,1); Rvx, Rvy, Rvz = _diff(Rx,1), _diff(Ry,1), _diff(Rz,1)
        Lvmag, Rvmag = _diff(Lmag,1), _diff(Rmag,1)
        Lax, Lay, Laz = _diff(Lx,2), _diff(Ly,2), _diff(Lz,2); Rax, Ray, Raz = _diff(Rx,2), _diff(Ry,2), _diff(Rz,2)
        Lamag, Ramag = _diff(Lmag,2), _diff(Rmag,2)

        base_tag = f"PAIR_{base}"

        # координаты
        out.update(_pair_feats_dict(f"{base_tag}_x", Lx, Rx))
        out.update(_pair_feats_dict(f"{base_tag}_y", Ly, Ry))
        out.update(_pair_feats_dict(f"{base_tag}_z", Lz, Rz))
        out.update(_pair_feats_dict(f"{base_tag}_mag", Lmag, Rmag))

        # скорость/ускорение — только «энергетические» и corr
        out.update(_pair_feats_dict(f"{base_tag}_vmag", Lvmag, Rvmag))
        out.update(_pair_feats_dict(f"{base_tag}_amag", Lamag, Ramag))

        # для глобалок возьмём RMS(|L-R|)
        for key, dif in (("x", Lx-Rx), ("y", Ly-Ry), ("z", Lz-Rz), ("mag", Lmag-Rmag),
                         ("vmag", Lvmag-Rvmag), ("amag", Lamag-Ramag)):
            glob_abs_diff_rms[key].append(float(np.sqrt(np.mean(dif*dif))))

    # глобальные признаки асимметрии (средние по всем парам)
    if pairs:
        for k, arr in glob_abs_diff_rms.items():
            if arr:
                v = np.array(arr, dtype=np.float32)
                out[f"ASYM_global_{k}_diff_rms_mean"] = float(v.mean())
                out[f"ASYM_global_{k}_diff_rms_std"]  = float(v.std())
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

def filter_files_by_list(root: Path, basenames: Set[str], stems: Set[str], raw_paths: Set[str]) -> Tuple[list, int]:
    all_files = list(root.rglob("*.npy")) + list(root.rglob("*.NPY"))
    if not all_files:
        return [], 0
    basenames_l = set(basenames); stems_l = set(stems); raw_paths_l = set(raw_paths)
    matched = []; seen = set()
    for p in all_files:
        name_l = p.name.lower(); stem_l = p.stem.lower(); path_l = str(p.resolve()).lower()
        if (name_l in basenames_l) or (stem_l in stems_l) or (path_l in raw_paths_l):
            matched.append(p); seen.add(name_l); seen.add(stem_l)
    n_requested = len(basenames_l | stems_l | raw_paths_l)
    return sorted(matched), n_requested

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Экстрактор фич (статистика+динамика+L/R асимметрия) с фильтрацией по списку файлов")
    ap.add_argument("--data_dir", required=True, help="Папка с .npy (рекурсивно)")
    ap.add_argument("--out_csv",  required=True, help="Путь к итоговому CSV")
    ap.add_argument("--schema",   default=None,  help="schema_joints.json (опционально)")
    ap.add_argument("--filelist_csv", required=True, help="CSV с колонкой filename — какие файлы обрабатывать")
    ap.add_argument("--filelist_col", default="filename", help="Имя колонки со списком файлов (по умолчанию 'filename')")
    ap.add_argument("--max_len", type=int, default=0, help="макс. число кадров; 0 = без ограничения")
    ap.add_argument("--frame_pick", choices=["uniform","head","tail","center","random"], default="uniform", help="стратегия отбора кадров при обрезке")
    ap.add_argument("--len_slack", type=int, default=500, help="Сжимать только если T > max_len + len_slack")
    ap.add_argument("--encoding", default="utf-8-sig", help="Кодировка CSV файлов")
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

            feats = extract_features(A, names)
            feats.update({"file": str(p.resolve()), "basename": p.name, "stem": p.stem})
            rows.append(feats)
        except Exception as e:
            print(f"[skip] {p}: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("[err] Не удалось извлечь ни одной строки фичей.")

    df = pd.DataFrame(rows)
    # n_frames больше не добавляем; упорядочим первые колонки
    first = [c for c in ["file", "basename", "stem", "n_joints", "lr_pairs_count"] if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[done] Saved: {out_csv}  |  rows={len(df)}  cols={len(df.columns)}")

if __name__ == "__main__":
    main()
