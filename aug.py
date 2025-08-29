#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

def map_label(v):
    """Мультикласс: предпочитаем int; поддержим бинарные текстовые метки для совместимости."""
    if pd.isna(v): return None
    s = str(v).strip().lower()
    # чистое число?
    try:
        iv = int(float(s))
        return iv
    except Exception:
        pass
    # fallback для yes/no
    pos = {"1","yes","true","inj","injury","y","t"}
    neg = {"0","no","false","noinj","no injury","n","f"}
    if s in pos: return 1
    if s in neg: return 0
    return None

def choose_col(df: pd.DataFrame, preferred: str|None, pool: tuple[str,...], err: str) -> str:
    if preferred and preferred in df.columns: return preferred
    low = {c.lower(): c for c in df.columns}
    if preferred and preferred.lower() in low: return low[preferred.lower()]
    for c in pool:
        if c in df.columns: return c
        if c.lower() in low: return low[c.lower()]
    raise SystemExit(err)

def build_ci_index(data_dir: Path) -> dict[str, Path]:
    """case-insensitive: lower(stem) -> Path"""
    idx = {}
    for p in list(data_dir.rglob("*.npy")) + list(data_dir.rglob("*.NPY")):
        idx[p.stem.lower()] = p
    return idx

def load_len(npy_path: Path) -> int:
    A = np.load(npy_path, allow_pickle=False)
    return A.shape[0]

def split_chunks(A: np.ndarray, k: int) -> list[np.ndarray]:
    T = A.shape[0]
    if k <= 1: return [A]
    cuts = [round(i*T/k) for i in range(k+1)]
    return [A[cuts[i]:cuts[i+1]] for i in range(k) if cuts[i+1] > cuts[i]]

def parse_only_labels(s: str|None, labels_in_df: list[int]) -> set[int]:
    """'all' -> все классы из данных; '0,2,3' -> заданные; None -> только класс 0 (backward compat)."""
    if s is None:  # совместимость со старым поведением (резать 0)
        return {0}
    s = str(s).strip().lower()
    if s in ("all","*"):
        return set(labels_in_df)
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        out.add(int(tok))
    return out

def main():
    ap = argparse.ArgumentParser(description="Аугментация .npy: копирование, резка длинных рядов по классам (мультикласс)")
    ap.add_argument("--data_dir", required=True, help="Где лежат исходные .npy (рекурсивно)")
    ap.add_argument("--data_csv",  required=True, help="CSV с колонками filename, label (или укажи через --fname_col/--label_col)")
    ap.add_argument("--out_dir",   default="/kaggle/working", help="Куда складывать результат")
    ap.add_argument("--out_npy_subdir", default="npy_all", help="Подпапка с .npy внутри out_dir")
    ap.add_argument("--label_col", default=None, help="Имя колонки метки")
    ap.add_argument("--fname_col", default=None, help="Имя колонки файла")
    # новое:
    ap.add_argument("--only_labels", default="all",
                    help="Какие классы резать: 'all' или список через запятую (например '0,1,2,3'). По умолчанию all.")
    ap.add_argument("--only_label", type=int, default=None,
                    help="[DEPRECATED] Старый флаг для бинарного случая; используйте --only_labels")
    ap.add_argument("--min_frames", type=int, default=9000, help="Порог длины T для резки")
    ap.add_argument("--chunks", type=int, default=3, help="Сколько частей делать")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_npy  = out_dir / args.out_npy_subdir
    out_npy.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    label_col = choose_col(df, args.label_col,
                           ("label","No inj/ inj","no inj / inj","injury","target","y","class"),
                           "[err] не нашёл колонку метки — укажите --label_col")
    fname_col = choose_col(df, args.fname_col,
                           ("filename","file","path","basename","stem"),
                           "[err] не нашёл колонку имени файла — укажите --fname_col")

    df["_label"] = df[label_col].map(map_label)
    if df["_label"].isna().all():
        raise SystemExit("[err] не удалось привести метки к целым (0..K-1). Проверьте колонку метки.")

    # список классов в данных
    labels_present = sorted(df["_label"].dropna().astype(int).unique().tolist())

    # поддержка старого флага
    if args.only_label is not None:
        print("[warn] --only_label устарел; используйте --only_labels. Применяю only_labels из --only_label.")
        args.only_labels = str(args.only_label)

    to_cut = parse_only_labels(args.only_labels, labels_present)
    unknown = to_cut - set(labels_present)
    if unknown:
        print(f"[warn] В --only_labels есть классы, отсутствующие в данных: {sorted(unknown)}")

    # индекс исходных npy (case-insensitive)
    idx = build_ci_index(data_dir)
    if not idx:
        raise SystemExit(f"[err] в {data_dir} не найдено .npy")

    # статистика до
    before_cnt = Counter(df["_label"].dropna().astype(int).tolist())
    print("[stats] before:", dict(sorted(before_cnt.items())))

    # чтобы не копировать один и тот же файл многократно
    copied = set()
    new_rows = []
    miss = 0
    split_cnt = 0
    copy_cnt = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="process"):
        r = dict(zip(df.columns, row))
        lab = r["_label"]
        try:
            lab = int(lab)
        except Exception:
            lab = None
        stem = Path(str(r[fname_col])).stem.lower()  # case-insensitive
        src = idx.get(stem)
        if src is None:
            miss += 1
            r[fname_col] = Path(str(r[fname_col])).name
            new_rows.append(r)
            continue

        try:
            T = load_len(src)
        except Exception:
            T = -1

        if (lab is not None) and (lab in to_cut) and (T > args.min_frames > 0):
            A = np.load(src, allow_pickle=False)
            chunks = split_chunks(A, args.chunks)
            base = src.stem
            for i, ch in enumerate(chunks, 1):
                out_name = f"{base}_chunk{i}.npy"
                out_path = out_npy / out_name
                if not out_path.exists():
                    np.save(out_path, ch)
                rr = r.copy()
                rr[fname_col] = out_name
                new_rows.append(rr)
                split_cnt += 1
        else:
            out_name = src.name
            out_path = out_npy / out_name
            if out_name.lower() not in copied:
                if not out_path.exists():
                    shutil.copy2(src, out_path)
                copied.add(out_name.lower())
                copy_cnt += 1
            r[fname_col] = out_name
            new_rows.append(r)

    new_df = pd.DataFrame(new_rows)
    new_df.drop(columns=["_label"], inplace=True, errors="ignore")

    out_csv = out_dir / "data_split.csv"
    new_df.to_csv(out_csv, index=False)

    # статистика после
    after_cnt = Counter(pd.to_numeric(new_df[label_col], errors="coerce").dropna().astype(int).tolist())
    print("[stats] after:", dict(sorted(after_cnt.items())))

    print(f"\n[done] CSV: {out_csv}  | rows={len(new_df)}")
    print(f"[info] npy saved to: {out_npy}")
    print(f"[info] copied originals: {copy_cnt}  | generated chunks: {split_cnt}")
    if miss:
        print(f"[warn] не найдено файлов по CSV (регистр/имя): {miss}")

if __name__ == "__main__":
    main()
