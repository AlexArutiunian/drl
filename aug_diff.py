#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сильные офлайн-аугментации для временных рядов (.npy -> .npy), с сохранением различимости train/test по origin.

— Читает CSV ТОЛЬКО для train (после сплита).
— Делает ровно ×K (по умолчанию ×5): оригинал + (K-1) аугментаций.
— Сохраняет новые файлы как <stem>_aug##.npy в отдельной папке и CSV с колонками [filename, label, origin].
— origin восстанавливается из исходного имени и НЕ меняется (по нему можно делать групповой сплит без утечек).
— Без внешних зависимостей (NumPy). Если установлен tsaug — использует его дополнительно.
"""
from __future__ import annotations
import os, argparse, math, json, re
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============== вспомогательное ===============

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_seq(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim >= 2:
        t_axis = int(np.argmax(a.shape))
        if t_axis != 0:
            a = np.moveaxis(a, t_axis, 0)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
    return a.astype(np.float32, copy=False)

# origin: базовое имя без служебных суффиксов _chunk### / _### / _aug###
ORIGIN_RX = re.compile(r"(?:_chunk\d+|_aug\d+|_\d+)$", re.IGNORECASE)

def to_origin(stem: str) -> str:
    s = stem.lower()
    # срезаем по одному суффиксу пока есть
    while True:
        new = ORIGIN_RX.sub("", s)
        if new == s:
            return new
        s = new

# =============== сильные аугментации (NumPy) ===============

rng_global: np.random.Generator | None = None

def _rng() -> np.random.Generator:
    global rng_global
    if rng_global is None:
        rng_global = np.random.default_rng(42)
    return rng_global

# — jitter (gaussian noise)
def aug_jitter(x: np.ndarray, sigma_range=(0.005, 0.05)) -> np.ndarray:
    T, F = x.shape
    sigma = _rng().uniform(*sigma_range)
    return x + _rng().normal(0.0, sigma, size=(T, F)).astype(np.float32)

# — scaling per-feature
def aug_scale(x: np.ndarray, scale_range=(-0.2, 0.2)) -> np.ndarray:
    T, F = x.shape
    s = 1.0 + _rng().uniform(scale_range[0], scale_range[1], size=(1, F)).astype(np.float32)
    return x * s

# — time shift (circular)
def aug_shift(x: np.ndarray, max_frac=0.1) -> np.ndarray:
    T, _ = x.shape
    k = int(_rng().integers(-int(T*max_frac), int(T*max_frac)+1))
    if k == 0: return x
    return np.roll(x, k, axis=0)

# — time masking (zero windows)
def aug_time_mask(x: np.ndarray, max_frac=0.1) -> np.ndarray:
    T, F = x.shape
    w = int(_rng().integers(max(1, T//50), max(2, int(T*max_frac))))
    t0 = int(_rng().integers(0, max(1, T-w)))
    y = x.copy()
    y[t0:t0+w] = 0.0
    return y

# — magnitude warp: плавная кривая-множитель по времени

def _rand_curve(T: int, knots: int = 4, scale: float = 0.2) -> np.ndarray:
    # узлы иконические, по ним генерим коэффициенты, далее интерполяция
    xs = np.linspace(0, T-1, num=knots).astype(np.float32)
    ys = 1.0 + _rng().normal(0.0, scale, size=(knots,)).astype(np.float32)
    base = np.interp(np.arange(T, dtype=np.float32), xs, ys)
    return base

def aug_mag_warp(x: np.ndarray, knots: int = 4, scale: float = 0.2) -> np.ndarray:
    T, F = x.shape
    m = np.stack([_rand_curve(T, knots, scale) for _ in range(F)], axis=1)
    return x * m.astype(np.float32)

# — time warp: нелинейная карта времени + ресемплинг обратно в T

def aug_time_warp(x: np.ndarray, knots: int = 4, sigma: float = 0.2) -> np.ndarray:
    T, F = x.shape
    # случайная монотонная карта времени: интеграл от положительного шума
    dv = np.abs(_rng().normal(1.0, sigma, size=(knots,)).astype(np.float32))
    xs = np.linspace(0, 1, num=knots, dtype=np.float32)
    v = np.interp(np.linspace(0,1,T, dtype=np.float32), xs, dv)
    tmap = np.cumsum(v)
    tmap = (tmap - tmap[0]) / (tmap[-1] - tmap[0] + 1e-12)  # [0,1]
    src_idx = tmap * (T-1)
    i0 = np.floor(src_idx).astype(np.int32)
    i1 = np.clip(i0+1, 0, T-1)
    a = (src_idx - i0).astype(np.float32)[..., None]
    y = (1-a)*x[i0] + a*x[i1]
    return y.astype(np.float32)

# — permute small segments (3..6)

def aug_permute(x: np.ndarray, nseg_range=(3,6)) -> np.ndarray:
    T, F = x.shape
    n = int(_rng().integers(nseg_range[0], nseg_range[1]+1))
    cuts = np.linspace(0, T, n+1, dtype=int)
    order = _rng().permutation(n)
    segs = [x[cuts[i]:cuts[i+1]] for i in order]
    return np.concatenate(segs, axis=0).astype(np.float32)

# Композиция: случайный выбор 3–5 трансформаций и случайный порядок
AUG_POOL = [aug_jitter, aug_scale, aug_shift, aug_time_mask, aug_mag_warp, aug_time_warp, aug_permute]

def strong_augment(x: np.ndarray) -> np.ndarray:
    k = int(_rng().integers(3, 6))  # 3..5
    funcs = _rng().choice(AUG_POOL, size=k, replace=False)
    y = x.copy()
    for f in funcs:
        y = f(y)
    return y

# (Опционально) подключим tsaug, если установлен
try:
    from tsaug import TimeWarp as TS_TimeWarp, Drift as TS_Drift, AddNoise as TS_AddNoise, Dropout as TS_Dropout
    HAS_TSAUG = True
except Exception:
    HAS_TSAUG = False


def tsaug_augment_batch(batch: np.ndarray) -> np.ndarray:
    # batch: (N,T,F)
    if not HAS_TSAUG:
        return np.stack([strong_augment(x) for x in batch], axis=0)
    aug = (TS_TimeWarp() + TS_Drift(max_drift=(0.02, 0.2)) @ 0.8 + TS_AddNoise(scale=(0.01, 0.06)) @ 0.8 + TS_Dropout(p=0.03, fill=0) @ 0.5)
    return aug.augment(batch)

# =============== основная логика ===============

def main():
    ap = argparse.ArgumentParser(description="Сильные аугментации временных рядов для train: делаем ×K и сохраняем CSV")
    ap.add_argument("--train_csv", required=True, help="CSV после группового сплита (ТОЛЬКО train) со столбцами filename,label")
    ap.add_argument("--data_dir",  required=True, help="Корень исходных .npy")
    ap.add_argument("--out_dir",   required=True, help="Куда писать (папки и CSV)")
    ap.add_argument("--out_npy_subdir", default="npy_aug", help="Подпапка для .npy внутри out_dir")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col",    default="label")
    ap.add_argument("--times", type=int, default=5, help="Во сколько раз увеличить датасет (включая оригинал)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0, help="Ограничить число строк для отладки (0=все)")
    args = ap.parse_args()

    _ = _rng()  # инициализация
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_npy  = out_dir / args.out_npy_subdir
    ensure_dir(out_npy)

    df = pd.read_csv(args.train_csv)
    if args.filename_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"[err] CSV должен содержать колонки '{args.filename_col}' и '{args.label_col}'")

    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    n_aug = max(0, args.times - 1)
    if n_aug == 0:
        print("[info] times=1 -> просто копируем оригиналы в новую папку и пишем CSV")

    out_rows = []
    missing = 0
    total = len(df)

    for r in tqdm(df.itertuples(index=False), total=total, desc="augment"):
        fname = getattr(r, args.filename_col)
        label = getattr(r, args.label_col)
        src = data_dir / str(fname)
        if not src.exists():
            # попробуем без путей/подкаталогов — только по имени
            alt = list(data_dir.rglob(Path(fname).name))
            if len(alt):
                src = alt[0]
        if not src.exists():
            missing += 1
            continue

        x = np.load(src, allow_pickle=False)
        x = sanitize_seq(x)
        T, F = x.shape

        stem = Path(fname).stem
        origin = to_origin(stem)

        # 1) оригинал
        base_out = out_npy / f"{stem}.npy"
        if not base_out.exists():
            np.save(base_out, x)
        out_rows.append({args.filename_col: base_out.name, args.label_col: label, "origin": origin})

        # 2) аугментации
        if n_aug > 0:
            # батч для tsaug, если есть, иначе поштучно strong_augment
            if HAS_TSAUG:
                batch = np.repeat(x[None, ...], n_aug, axis=0)
                aug_batch = tsaug_augment_batch(batch)
                for k in range(n_aug):
                    y = aug_batch[k]
                    out_name = f"{stem}_aug{(k+1):02d}.npy"
                    np.save(out_npy / out_name, y)
                    out_rows.append({args.filename_col: out_name, args.label_col: label, "origin": origin})
            else:
                for k in range(n_aug):
                    y = strong_augment(x)
                    out_name = f"{stem}_aug{(k+1):02d}.npy"
                    np.save(out_npy / out_name, y)
                    out_rows.append({args.filename_col: out_name, args.label_col: label, "origin": origin})

    out_df = pd.DataFrame(out_rows, columns=[args.filename_col, args.label_col, "origin"])

    # sanity: проверить, что на каждую исходную строку вышло ровно times записей
    counts = out_df.groupby("origin").size().describe()
    print("[stats] augmented per origin (describe):\n", counts)

    out_csv = out_dir / "train_augmented.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n[done] rows_in={len(df)}  rows_out={len(out_df)} (target≈{len(df)*args.times})")
    if missing:
        print(f"[warn] не найдено файлов: {missing}")
    print(f"npy saved to: {out_npy}")
    print(f"CSV saved to: {out_csv}")

if __name__ == "__main__":
    main()
