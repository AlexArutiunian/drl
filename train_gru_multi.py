#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRU-мультиклассификатор поверх последовательностей из .npy
— совместим по интерфейсу с вашим бинарным вариантом, но поддерживает K классов
— аккуратный сплит, нормализация, сохранение метрик/артефактов и красивые графики
"""
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import os, json, argparse, math, random, re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # headless сохранение графиков
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ====================== ПЛОТЫ ======================
def save_history_plots(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for metric in ["loss", "accuracy"]:
        train = history.history.get(metric, [])
        val   = history.history.get(f"val_{metric}", [])
        if not train:
            continue
        plt.figure()
        plt.plot(range(1, len(train)+1), train, label=f"train_{metric}")
        if val: plt.plot(range(1, len(val)+1), val, label=f"val_{metric}")
        plt.xlabel("Epoch"); plt.ylabel(metric); plt.legend()
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=150)
        plt.close()

def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix (test)"):
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_roc_ovr(y_true: np.ndarray, prob: np.ndarray, class_names: List[str], out_path: str, title: str = "ROC (OVR)"):
    # y_true: int (0..K-1), prob: [N,K]
    K = prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(K)))  # [N,K]
    fpr, tpr, roc_auc = {}, {}, {}
    valid = []
    for i in range(K):
        yi = y_bin[:, i]
        if yi.sum() == 0 or yi.sum() == len(yi):  # класс отсутствует или только он один
            continue
        fi, ti, _ = roc_curve(yi, prob[:, i])
        fpr[i], tpr[i] = fi, ti
        roc_auc[i] = auc(fi, ti)
        valid.append(i)
    if not valid:
        return
    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(6,5), dpi=150)
    for i in valid:
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.9, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(all_fpr, mean_tpr, lw=2.0, label=f"macro-average (AUC={macro_auc:.3f})")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(title)
    plt.legend(fontsize=8, loc="lower right", ncol=1, frameon=True)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def save_confidence_buckets_multiclass(prob, y_true, y_pred, out_path, model_name="gru-multiclass"):
    """Уверенность = top-1 вероятность. Три панели: все, корректные, ошибочные."""
    prob = np.asarray(prob, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    conf = prob.max(axis=1)
    bins = [0.0, 0.5, 0.6, 0.8, 1.0001]
    labels = ["conf <50%", "50–60%", "60–80%", ">80%"]
    def bucket_counts(mask):
        if mask.sum() == 0:
            return np.zeros(len(labels), dtype=int)
        h, _ = np.histogram(conf[mask], bins=bins); return h
    panels = [
        (np.ones_like(conf, dtype=bool),        f"{model_name} — ALL"),
        (y_pred == y_true,                      f"{model_name} — CORRECT"),
        (y_pred != y_true,                      f"{model_name} — WRONG"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=150)
    for ax, (mask, title) in zip(axs, panels):
        cnts = bucket_counts(mask)
        bars = ax.barh(range(len(labels)), cnts, height=0.55)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Count")
        xmax = max(int(cnts.max()), 1)
        ax.set_xlim(0, xmax * 1.15)
        ax.bar_label(bars, labels=[str(int(v)) for v in cnts], padding=3, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

# ====================== УТИЛИТЫ ======================
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def sanitize_seq(a: np.ndarray) -> np.ndarray:
    """
    Приводит произвольный массив к форме (T, F):
    - заменяет NaN/Inf на 0
    - выбирает самую длинную ось как время и переносит её на 0
    - остальные оси сплющиваются в признаки
    """
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

# ----- чтение индекса и меток -----

def normalize_label(v) -> Optional[int]:
    """Принимает строку/число. Возвращает int-метку, если удаётся (поддержка 0..K-1 или 1..K)."""
    if v is None:
        return None
    s = str(v).strip().lower()
    # часто встречающиеся текстовые варианты: можно расширить при желании
    if s in ("no injury", "no_injury", "none", "neg", "negative"): return 0
    if s in ("injury", "pos", "positive"): return 1
    # числа
    try:
        f = float(s)
        if math.isfinite(f):
            return int(round(f))
    except Exception:
        return None
    return None


def build_items(csv_path: str, data_dir: str,
                filename_col="filename", label_col="No inj/ inj",
                debug_index: bool=False) -> Tuple[List[str], List[int], Dict[str,int], List[Tuple[str,str]]]:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    if fn_col not in df.columns:
        for c in df.columns:
            if "file" in c.lower() or "name" in c.lower():
                fn_col = c; break
    lb_col = cols.get(label_col.lower(), label_col)
    if lb_col not in df.columns:
        for c in df.columns:
            if any(k in c.lower() for k in ("inj", "label", "target", "class")):
                lb_col = c; break

    print(f"Using columns: filename_col='{fn_col}', label_col='{lb_col}'")

    items_x, items_y = [], []
    stats = {"ok":0, "no_file":0, "bad_label":0, "too_short":0, "error":0}
    skipped_examples = []

    def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
        rel = (rel or "").strip().replace("\\", "/").lstrip("/")
        cands: List[str] = []
        def push(x):
            if x not in cands:
                cands.append(x)
        push(os.path.join(data_dir, rel))
        if not rel.endswith(".npy"): push(os.path.join(data_dir, rel + ".npy"))
        if rel.endswith(".json"):
            b = rel[:-5]; push(os.path.join(data_dir, b + ".npy")); push(os.path.join(data_dir, rel + ".npy"))
        if rel.endswith(".json.npy"):
            push(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))
        b = os.path.basename(rel)
        push(os.path.join(data_dir, b))
        if not b.endswith(".npy"): push(os.path.join(data_dir, b + ".npy"))
        if b.endswith(".json"):
            push(os.path.join(data_dir, b[:-5] + ".npy")); push(os.path.join(data_dir, b + ".npy"))
        if b.endswith(".json.npy"):
            push(os.path.join(data_dir, b.replace(".json.npy", ".npy")))
        return cands

    def pick_existing_path(cands: List[str]) -> Optional[str]:
        for p in cands:
            if os.path.exists(p):
                return p
        return None

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
        rel = str(row.get(fn_col, "")).strip()
        lab_raw = row.get(lb_col, None)
        lab = normalize_label(lab_raw)

        status = ""; resolved = None; shape_txt = ""

        if not rel:
            stats["no_file"] += 1; status = "empty-filename"
            if len(skipped_examples)<10: skipped_examples.append((status, str(row.to_dict())))
        elif lab is None:
            stats["bad_label"] += 1; status = f"bad-label:{lab_raw}"
            if len(skipped_examples)<10: skipped_examples.append((status, rel))
        else:
            path = pick_existing_path(possible_npy_paths(data_dir, rel))
            if not path:
                stats["no_file"] += 1; status = "not-found"
                if len(skipped_examples)<10: skipped_examples.append((status, rel))
            else:
                try:
                    arr = np.load(path, allow_pickle=False, mmap_mode="r")
                    x = sanitize_seq(arr)
                    if x.ndim != 2 or x.shape[0] < 2:
                        stats["too_short"] += 1; status = "too-short"
                        if len(skipped_examples)<10: skipped_examples.append((status, os.path.basename(path)))
                    else:
                        items_x.append(path)
                        items_y.append(int(lab))
                        stats["ok"] += 1; status = "OK"; resolved = path
                        shape_txt = f"shape={tuple(x.shape)}"
                except Exception as e:
                    stats["error"] += 1; status = f"np-load:{type(e).__name__}"
                    if len(skipped_examples)<10: skipped_examples.append((status, os.path.basename(path) if path else rel))

        if debug_index:
            print(f"[{i:05d}] csv='{rel}' | label_raw='{lab_raw}' -> {status}" + (f" | path='{resolved}' {shape_txt}" if resolved else ""))
    print(stats)
    return items_x, items_y, stats, skipped_examples

# ======= статистики по длинам/нормализация =======
def probe_stats(items: List[Tuple[str,int]], downsample: int = 1, pctl: int = 95):
    lengths = []; feat = None
    for p, _ in items:
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)
        L = int(x.shape[0] // max(1, downsample))
        if x.ndim != 2 or L < 1:
            continue
        if feat is None: feat = int(x.shape[1])
        if L > 0: lengths.append(L)
    max_len = int(np.percentile(lengths, pctl)) if lengths else 256
    return max_len, (feat or 1)


def compute_norm_stats(items: List[Tuple[str,int]], feat_dim: int, downsample: int, max_len_cap: int, sample_items: int = 512):
    rng = random.Random(42)
    pool = items if len(items) <= sample_items else rng.sample(items, sample_items)
    count = 0
    mean = np.zeros(feat_dim, dtype=np.float64)
    m2   = np.zeros(feat_dim, dtype=np.float64)
    for p, _ in pool:
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)
        x = x[::max(1, downsample)]
        if x.shape[0] > max_len_cap: x = x[:max_len_cap]
        for t in range(x.shape[0]):
            count += 1
            delta = x[t] - mean
            mean += delta / max(count, 1)
            m2   += delta * (x[t] - mean)
    var = m2 / max(1, (count - 1))
    std = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)
    return mean.astype(np.float32), std

# ==================== TF МОДЕЛЬ ====================
def lazy_tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    return tf, layers, models


def make_datasets(items, labels, max_len, feat_dim, bs, downsample, mean, std, replicas):
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE

    def gen(indices):
        def _g():
            for i in indices:
                p = items[i]; y = labels[i]
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                x = sanitize_seq(arr)
                x = x[::max(1, downsample)]
                if x.shape[0] < 2: continue
                if x.shape[0] > max_len: x = x[:max_len]
                x = (x - mean) / std
                yield x, np.int32(y)
        return _g

    sig = (
        tf.TensorSpec(shape=(None, feat_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    def pad_map(x, y):
        T = tf.shape(x)[0]
        pad_t = tf.maximum(0, max_len - T)
        x = tf.pad(x, [[0, pad_t], [0, 0]])
        x = x[:max_len]
        x.set_shape([max_len, feat_dim])
        return x, y

    def make(indices, shuffle=False, drop_remainder=False):
        ds = tf.data.Dataset.from_generator(gen(indices), output_signature=sig)
        if shuffle:
            ds = ds.shuffle(4096, reshuffle_each_iteration=True)
        ds = ds.map(pad_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(bs, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    return make


def build_gru_model(input_shape, num_class: int, learning_rate=1e-3, mixed_precision=False):
    tf, layers, models = lazy_tf()
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_class, activation="softmax", dtype="float32")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# заглушка контекста, если нет стратегии
class contextlib_null:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

# ==================== MAIN ====================
def main():
    ap = argparse.ArgumentParser("GRU multiclass classifier over NPY sequences")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="No inj/ inj")
    ap.add_argument("--label_index_csv", default=None, help="(опц.) справочник меток: label,HumanName (label: 1..K или 0..K-1)")
    ap.add_argument("--out_dir", default="output_run_gru_multiclass")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max_len", default="auto")  # auto -> 95 перцентиль
    ap.add_argument("--gpus", default="all")      # "all" | "cpu" | "0,1"
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--print_csv_preview", action="store_true")
    ap.add_argument("--debug_index", action="store_true")
    ap.add_argument("--peek", type=int, default=0)
    args = ap.parse_args()

    # Видимость GPU до импорта TF
    if args.gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.gpus.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Индекс
    items_x, items_y_raw, stats, skipped = build_items(
        args.csv, args.data_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        debug_index=args.debug_index
    )
    assert items_x and items_y_raw, "Не найдено валидных .npy и меток"

    # ===== маппинг меток к 0..K-1 (как в XGB-скрипте) =====
    raw_labels_sorted = sorted(set(int(v) for v in items_y_raw))
    if 0 in raw_labels_sorted:
        raw2zero = {int(v): int(v) for v in raw_labels_sorted}
    else:
        raw2zero = {int(v): int(v-1) for v in raw_labels_sorted}
    labels_all = np.array([raw2zero[int(v)] for v in items_y_raw], dtype=np.int32)
    items = list(zip(items_x, labels_all))

    if args.print_csv_preview:
        print("\n=== CSV preview (first 5 rows) ===")
        df_preview = pd.read_csv(args.csv)
        print(df_preview.head(5).to_string(index=False))
        if args.label_col in df_preview.columns:
            print(f"\nLabel value counts in '{args.label_col}':")
            print(df_preview[args.label_col].value_counts(dropna=False))

    if args.peek > 0:
        print(f"\n=== Peek first {min(args.peek, len(items))} matched items ===")
        for (pth, lab) in items[:args.peek]:
            try:
                arr = np.load(pth, allow_pickle=False, mmap_mode="r")
                x = sanitize_seq(arr)
                print(f"OK  label={lab} | {pth} | shape={x.shape}")
            except Exception as e:
                print(f"FAIL to load for peek: {pth} -> {type(e).__name__}")

    # Статы по длине/размеру признака
    if str(args.max_len).strip().lower() == "auto":
        max_len, feat_dim = probe_stats(items, downsample=args.downsample, pctl=95)
        max_len = int(max(8, min(max_len, 20000)))
    else:
        max_len = int(args.max_len)
        aa = np.load(items_x[0], allow_pickle=False, mmap_mode="r")
        aa = sanitize_seq(aa)
        feat_dim = int(aa.shape[1])
    print(f"max_len={max_len} | feat_dim={feat_dim}")

    # Сплит 70/10/20 (стратифицированный)
    # ==== ГРУППОВОЙ сплит без утечек по префиксу ====
    import re
    from sklearn.model_selection import GroupShuffleSplit
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        HAS_SGKF = True
    except Exception:
        HAS_SGKF = False

    def to_origin(path_or_name: str) -> str:
        b = os.path.basename(str(path_or_name))
        st = os.path.splitext(b)[0].lower()
        # убираем суффиксы чанков: _chunk123 или _123
        st = re.sub(r"(?:_chunk\d+|_\d+)$", "", st)
        return st

    groups_all = np.array([to_origin(p) for p in items_x])
    y_all = labels_all
    idx_all = np.arange(len(items_x))

    # сначала ~20% в test, затем из train ~12.5% в dev (итого 70/10/20)
    if HAS_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)   # 1/5 ≈ 20%
        tr_idx, te_idx = next(sgkf.split(idx_all, y_all, groups_all))
        sgkf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=123) # 1/8 ≈ 12.5%
        tr2_rel, dev_rel = next(sgkf2.split(tr_idx, y_all[tr_idx], groups_all[tr_idx]))
        idx_train = tr_idx[tr2_rel]
        idx_dev   = tr_idx[dev_rel]
        idx_test  = te_idx
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(idx_all, y_all, groups_all))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125/0.80, random_state=43)  # ≈10% от всего
        tr2_idx, dev_rel = next(gss2.split(tr_idx, y_all[tr_idx], groups_all[tr_idx]))
        idx_train = tr_idx[tr2_idx]
        idx_dev   = tr_idx[dev_rel]
        idx_test  = te_idx

    # проверки на утечки групп
    inter_train_test = set(groups_all[idx_train]) & set(groups_all[idx_test])
    inter_dev_test   = set(groups_all[idx_dev])   & set(groups_all[idx_test])
    inter_train_dev  = set(groups_all[idx_train]) & set(groups_all[idx_dev])
    assert not inter_train_test, f"LEAK: общие префиксы между train и test: {sorted(list(inter_train_test))[:5]} ..."
    assert not inter_dev_test,   f"LEAK: общие префиксы между dev и test: {sorted(list(inter_dev_test))[:5]} ..."
    assert not inter_train_dev,  f"LEAK: общие префиксы между train и dev: {sorted(list(inter_train_dev))[:5]} ..."

    # в тесте не должно быть классов, которых нет в train
    train_classes = np.unique(y_all[idx_train])
    mask_te_keep = np.isin(y_all[idx_test], train_classes)
    if mask_te_keep.sum() < len(idx_test):
        idx_test = idx_test[mask_te_keep]

    print(f"[split] train={len(idx_train)} dev={len(idx_dev)} test={len(idx_test)} | "
        f"groups: train={len(set(groups_all[idx_train]))} dev={len(set(groups_all[idx_dev]))} "
        f"test={len(set(groups_all[idx_test]))}")

    # создаём папку и сохраняем аудит сплита
    ensure_dir(args.out_dir)
    split_df = pd.DataFrame({
        "path":   [items_x[i] for i in np.concatenate([idx_train, idx_dev, idx_test])],
        "origin": [groups_all[i] for i in np.concatenate([idx_train, idx_dev, idx_test])],
        "label":  [int(y_all[i]) for i in np.concatenate([idx_train, idx_dev, idx_test])],
        "split":  (["train"]*len(idx_train) + ["dev"]*len(idx_dev) + ["test"]*len(idx_test))
    })
    split_df.to_csv(os.path.join(args.out_dir, "split_groups.csv"), index=False)


    # Норм-статы по train
    mean, std = compute_norm_stats([items[i] for i in idx_train], feat_dim, args.downsample, max_len)
    np.savez_compressed(os.path.join(args.out_dir, "norm_stats.npz"), mean=mean, std=std, max_len=max_len)

    # TF / стратегия
    tf, layers, models = lazy_tf()
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass

    if args.mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")  # выходной Dense уже float32

    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None
    replicas = strategy.num_replicas_in_sync if strategy else 1
    global_bs = args.batch_size * replicas
    print(f"GPUs: {len(gpus)} | replicas: {replicas} | global_batch: {global_bs}")

    # Датасеты
    make_ds = make_datasets(items_x, labels_all, max_len, feat_dim, global_bs,
                            args.downsample, mean, std, replicas)
    train_ds = make_ds(idx_train, shuffle=True,  drop_remainder=(replicas > 1))
    dev_ds   = make_ds(idx_dev,   shuffle=False, drop_remainder=False)
    test_ds  = make_ds(idx_test,  shuffle=False, drop_remainder=False)

    # Число классов исходит из train
    classes_in_train = np.unique(labels_all[idx_train])
    num_class = int(classes_in_train.size)
    assert classes_in_train.min() == 0 and classes_in_train.max() == num_class - 1, \
        f"Train labels must be 0..K-1, got {classes_in_train}"
    print(f"[info] num_class used: {num_class}")

    # Веса классов (balanced)
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.arange(num_class, dtype=np.int32)
    w = compute_class_weight(class_weight="balanced", classes=cls, y=labels_all[idx_train])
    class_weight = {int(c): float(wi) for c, wi in zip(cls, w)}
    print("class_weight:", class_weight)

    # Строим модель
    ctx = strategy.scope() if strategy else contextlib_null()
    with ctx:
        model = build_gru_model((max_len, feat_dim), num_class=num_class, learning_rate=1e-3, mixed_precision=args.mixed_precision)

    # Коллбеки
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    # Обучение
    hist = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1
    )

    # Предсказания и метрики (multiclass)
    prob_dev  = model.predict(dev_ds,  verbose=0).astype(np.float32)  # [N,K]
    y_dev     = labels_all[idx_dev]
    pred_dev  = prob_dev.argmax(axis=1)

    prob_test = model.predict(test_ds, verbose=0).astype(np.float32)
    y_test    = labels_all[idx_test]
    pred_test = prob_test.argmax(axis=1)

    # Метрики (macro/micro F1 + accuracy)
    acc = accuracy_score(y_test, pred_test)
    f1_macro = f1_score(y_test, pred_test, average="macro")
    f1_micro = f1_score(y_test, pred_test, average="micro")

    # Красивые имена классов
    target_names = [f"class_{i}" for i in range(num_class)]
    if args.label_index_csv and os.path.exists(args.label_index_csv):
        LI = pd.read_csv(args.label_index_csv)
        # предполагается колонка "label" (сырые метки 1..K или 0..K-1) и колонка с человеко-понятным именем
        name_col = next((c for c in ["InjuryClass", "name", "class", "label_name", "title"] if c in LI.columns), None)
        if name_col is not None and "label" in LI.columns:
            # построим обратную маппу zero->raw
            zero2raw = {z:r for r,z in raw2zero.items()}
            LI["_zero"] = LI["label"].astype(int).map({r:z for z,r in zero2raw.items()})
            LI = LI[LI["_zero"].notna()].sort_values("_zero")
            names = LI[name_col].astype(str).tolist()
            if len(names) == num_class:
                target_names = names

    report_txt = classification_report(y_test, pred_test, target_names=target_names, digits=3)
    print("\n=== TEST ===")
    print(report_txt)

    # Плоты
    save_history_plots(hist, args.out_dir)
    cm = confusion_matrix(y_test, pred_test)
    save_confusion_matrix(cm, target_names, os.path.join(args.out_dir, "cm.png"))
    try:
        save_roc_ovr(y_test, prob_test, target_names, os.path.join(args.out_dir, "roc_ovr.png"))
    except Exception:
        pass
    save_confidence_buckets_multiclass(prob_test, y_test, pred_test, os.path.join(args.out_dir, "confidence_buckets.png"), model_name="gru-multiclass")

    # Сохранения
    model.save(os.path.join(args.out_dir, "model.keras"))
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "n_test": int(len(y_test)),
            "classes": target_names,
            "per_class": classification_report(y_test, pred_test, target_names=target_names, digits=4, output_dict=True)
        }, f, ensure_ascii=False, indent=2)

    # CSV с предсказаниями (в человеческие метки, если удастся восстановить)
    zero2raw = {z:r for r,z in raw2zero.items()}
    test_df = pd.DataFrame({
        "y_true": [zero2raw.get(int(v), int(v)) for v in y_test],
        "y_pred": [zero2raw.get(int(v), int(v)) for v in pred_test],
        "y_pred_topprob": prob_test.max(axis=1)
    })
    for j in range(prob_test.shape[1]):
        human_lbl = zero2raw.get(j, j)
        test_df[f"proba_{human_lbl}"] = prob_test[:, j]
    test_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    # Также сохраним маппинг меток
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"raw_to_zero_based": raw2zero, "classes_in_train": sorted(map(int, classes_in_train))}, f, ensure_ascii=False, indent=2)

    print("Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
