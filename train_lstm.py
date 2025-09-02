#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

python lstm_multiclass.py \
  --csv /kaggle/working/aug_multi/data_split.csv \
  --data_dir /kaggle/working/aug_multi/npy_all \
  --label_col label \
  --filename_col filename \
  --epochs 30 --batch_size 16 --max_len auto --downsample 1

"""
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import os, json, argparse, math, random, re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_curve, auc
)

# =============== графики ===============
def save_history_plots(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for metric in ["loss", "accuracy"]:
        tr = history.history.get(metric, [])
        va = history.history.get(f"val_{metric}", [])
        if not tr: continue
        plt.figure()
        plt.plot(range(1, len(tr)+1), tr, label=f"train_{metric}")
        if va: plt.plot(range(1, len(va)+1), va, label=f"val_{metric}")
        plt.xlabel("Epoch"); plt.ylabel(metric); plt.legend()
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=150)
        plt.close()

def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)\
        .plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def save_roc_ovr(y_true: np.ndarray, prob: np.ndarray, class_names: List[str], out_path: str):
    # OVR ROC — рисуем если хотя бы для пары классов это имеет смысл
    K = prob.shape[1]
    # one-vs-rest
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(K)))
    fpr, tpr, roc_auc, valid = {}, {}, {}, []
    for i in range(K):
        yi = y_bin[:, i]
        if yi.sum()==0 or yi.sum()==len(yi):  # класс отсутствует в тесте
            continue
        fi, ti, _ = roc_curve(yi, prob[:, i])
        fpr[i], tpr[i] = fi, ti
        roc_auc[i] = auc(fi, ti)
        valid.append(i)
    if not valid: return
    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(6,5), dpi=150)
    for i in valid:
        plt.plot(fpr[i], tpr[i], lw=1, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(all_fpr, mean_tpr, lw=2.0, label=f"macro (AUC={macro_auc:.3f})")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path); plt.close()

# =============== утилиты ===============
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def sanitize_seq(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim >= 2:
        t_axis = int(np.argmax(a.shape))
        if t_axis != 0: a = np.moveaxis(a, t_axis, 0)
        if a.ndim > 2: a = a.reshape(a.shape[0], -1)
    return a.astype(np.float32, copy=False)

def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    cands = []
    def push(x): 
        if x not in cands: cands.append(x)
    push(os.path.join(data_dir, rel))
    if not rel.endswith(".npy"): push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json"):
        base = rel[:-5]; push(os.path.join(data_dir, base + ".npy")); push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json.npy"): push(os.path.join(data_dir, rel.replace(".json.npy",".npy")))
    b = os.path.basename(rel)
    push(os.path.join(data_dir, b))
    if not b.endswith(".npy"): push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json"):
        push(os.path.join(data_dir, b[:-5] + ".npy")); push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json.npy"): push(os.path.join(data_dir, b.replace(".json.npy",".npy")))
    return cands

def pick_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

# читаем CSV и строим список (путь, сырая_метка)
def build_items(csv_path: str, data_dir: str,
                filename_col="filename", label_col="label",
                debug_index: bool=False) -> Tuple[List[str], List[object], Dict[str,int], List[Tuple[str,str]]]:
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
            if any(k in c.lower() for k in ("inj","label","target","class")):
                lb_col = c; break

    print(f"Using columns: filename_col='{fn_col}', label_col='{lb_col}'")

    items_x, items_yraw = [], []
    stats = {"ok":0, "no_file":0, "bad_label":0, "too_short":0, "error":0}
    skipped = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
        rel = str(row.get(fn_col, "")).strip()
        lab_raw = row.get(lb_col, None)

        status=""; resolved=None; shape_txt=""
        if not rel:
            stats["no_file"] += 1; status="empty-filename"
            if len(skipped)<10: skipped.append((status, str(row.to_dict())))
            continue

        path = pick_existing_path(possible_npy_paths(data_dir, rel))
        if not path:
            stats["no_file"] += 1; status="not-found"
            if len(skipped)<10: skipped.append((status, rel))
            continue

        try:
            arr = np.load(path, allow_pickle=False, mmap_mode="r")
            x = sanitize_seq(arr)
            if x.ndim!=2 or x.shape[0]<2:
                stats["too_short"] += 1; status="too-short"
                if len(skipped)<10: skipped.append((status, os.path.basename(path)))
            else:
                items_x.append(path); items_yraw.append(lab_raw)
                stats["ok"] += 1; status="OK"; resolved=path; shape_txt=f"shape={tuple(x.shape)}"
        except Exception as e:
            stats["error"] += 1; status=f"np-load:{type(e).__name__}"
            if len(skipped)<10: skipped.append((status, os.path.basename(path) if path else rel))

        if debug_index:
            print(f"[{i:05d}] csv='{rel}' | label_raw='{lab_raw}' -> {status}" + (f" | path='{resolved}' {shape_txt}" if resolved else ""))

    print(stats)
    return items_x, items_yraw, stats, skipped

def probe_stats(items: List[Tuple[str,object]], downsample: int = 1, pctl: int = 95):
    lengths, feat = [], None
    for p, _ in items:
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)
        L = int(x.shape[0] // max(1, downsample))
        if x.ndim != 2 or L < 1: continue
        if feat is None: feat = int(x.shape[1])
        if L > 0: lengths.append(L)
    max_len = int(np.percentile(lengths, pctl)) if lengths else 256
    return max_len, feat or 1

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
            mean += delta / count
            m2   += delta * (x[t] - mean)
    var = m2 / max(1, (count - 1))
    std = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)
    return mean.astype(np.float32), std

# =============== TF часть ===============
def lazy_tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    return tf, layers, models

def make_datasets(paths, labels, max_len, feat_dim, bs, downsample, mean, std, replicas,
                  shuffle_buf=1024, prefetch_n=2, num_parallel_calls=2, clip_abs=8.0):
    """Память-дружелюбный пайплайн: шеффлим индексы, данные грузим лениво."""
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE
    labels = np.asarray(labels)

    def _load_one_py(i):
        i = int(i)
        p = paths[i]; y = labels[i]
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)                     # (T,F)
        x = x[::max(1, downsample)]
        if x.shape[0] > max_len: x = x[:max_len]
        x = (x - mean) / np.maximum(std, 1e-2)    # защита
        x = np.clip(x, -clip_abs, clip_abs)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32), np.int32(y)

    def _load_one_tf(i):
        x, y = tf.py_function(_load_one_py, [i], Tout=(tf.float32, tf.int32))
        x.set_shape([None, feat_dim]); y.set_shape([])
        return x, y

    def _pad_map(x, y):
        T = tf.shape(x)[0]
        pad_t = tf.maximum(0, max_len - T)
        x = tf.pad(x, [[0, pad_t], [0, 0]])[:max_len]
        x.set_shape([max_len, feat_dim])
        return x, y

    def make(indices, shuffle=False, drop_remainder=False):
        indices = np.asarray(indices, dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices(indices)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(indices), shuffle_buf),
                            reshuffle_each_iteration=True)
        ds = ds.map(_load_one_tf, num_parallel_calls=num_parallel_calls, deterministic=False)
        ds = ds.map(_pad_map,     num_parallel_calls=num_parallel_calls, deterministic=False)
        ds = ds.batch(bs, drop_remainder=drop_remainder)
        ds = ds.prefetch(prefetch_n)

        opts = tf.data.Options()
        opts.experimental_deterministic = False
        opts.threading.max_intra_op_parallelism = 1
        ds = ds.with_options(opts)
        return ds

    return make

def build_lstm_multiclass(input_shape, num_class: int, learning_rate=1e-3, mixed_precision=False):
    tf, layers, models = lazy_tf()
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_class, activation="softmax", dtype="float32")(x)
    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class contextlib_null:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

# =============== main ===============
def main():
    ap = argparse.ArgumentParser("LSTM multiclass classifier over NPY sequences")
    ap.add_argument("--out_dir", default="output_run_lstm_multi")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="label")   # укажи свою колонку
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max_len", default="auto")
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--print_csv_preview", action="store_true")
    ap.add_argument("--debug_index", action="store_true")
    ap.add_argument("--peek", type=int, default=0)
    args = ap.parse_args()

    # GPU vis
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

    # Маппинг сырых меток -> 0..K-1 (поддержка и чисел, и строк)
    def to_key(v):
        s = str(v).strip().lower()
        try:
            f = float(s)
            if math.isfinite(f):
                return int(round(f))
        except Exception:
            pass
        return s  # строковый класс

    raw_vals = [to_key(v) for v in items_y_raw]
    uniq_raw = list(dict.fromkeys(raw_vals))   # порядок появления
    raw2zero = {rv:i for i, rv in enumerate(sorted(uniq_raw, key=lambda x: (isinstance(x,str), x)))}
    labels_all = np.array([raw2zero[to_key(v)] for v in items_y_raw], dtype=np.int32)

    items = list(zip(items_x, labels_all))
    paths = items_x

    if args.print_csv_preview:
        print("\n=== CSV preview (first 5 rows) ===")
        df_preview = pd.read_csv(args.csv)
        print(df_preview.head(5).to_string(index=False))
        if args.label_col in df_preview.columns:
            print(f"\nLabel value counts in '{args.label_col}':")
            print(df_preview[args.label_col].value_counts(dropna=False))

    if args.peek > 0:
        for pth, lab in items[:args.peek]:
            try:
                arr = np.load(pth, allow_pickle=False, mmap_mode="r")
                x = sanitize_seq(arr)
                print(f"OK  label={lab} | {pth} | shape={x.shape}")
            except Exception as e:
                print(f"FAIL load: {pth} -> {type(e).__name__}")

    # Статы
    if str(args.max_len).strip().lower() == "auto":
        max_len, feat_dim = probe_stats(items, downsample=args.downsample, pctl=95)
        max_len = int(max(8, min(max_len, 20000)))
    else:
        max_len = int(args.max_len)
        aa = np.load(paths[0], allow_pickle=False, mmap_mode="r")
        aa = sanitize_seq(aa)
        feat_dim = int(aa.shape[1])
    print(f"max_len={max_len} | feat_dim={feat_dim}")

    # ===== групповой сплит без утечек =====
    from sklearn.model_selection import GroupShuffleSplit
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        HAS_SGKF = True
    except Exception:
        HAS_SGKF = False

    DF_SRC = pd.read_csv(args.csv)
    name_col = args.filename_col if args.filename_col in DF_SRC.columns else \
        next((c for c in DF_SRC.columns if c.lower().strip()==args.filename_col.lower()), None)
    origin_col = next((c for c in DF_SRC.columns if c.lower()=="origin"), None)

    csv_origin_map: Dict[str,str] = {}
    if name_col is not None and origin_col is not None:
        for n, o in zip(DF_SRC[name_col], DF_SRC[origin_col]):
            if pd.isna(n): continue
            csv_origin_map[os.path.basename(str(n)).lower()] = str(o).lower()

    ORX = re.compile(r'(?:_chunk\d+|_aug\d+|_\d+)$', re.IGNORECASE)
    def origin_from_name(path_or_name: str) -> str:
        st = os.path.splitext(os.path.basename(str(path_or_name)))[0].lower()
        while True:
            new = ORX.sub('', st)
            if new == st: break
            st = new
        return st

    groups_all = []
    for p in paths:
        key = os.path.basename(p).lower()
        g = csv_origin_map.get(key) or origin_from_name(key)
        groups_all.append(g)
    groups_all = np.array(groups_all)

    y_all = labels_all
    idx_all = np.arange(len(paths))
    if HAS_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, te_idx = next(sgkf.split(idx_all, y_all, groups_all))
        sgkf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=123)
        tr2_rel, dev_rel = next(sgkf2.split(tr_idx, y_all[tr_idx], groups_all[tr_idx]))
        idx_train = tr_idx[tr2_rel]; idx_dev = tr_idx[dev_rel]; idx_test = te_idx
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(idx_all, y_all, groups_all))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125/0.80, random_state=43)
        tr2_idx, dev_rel = next(gss2.split(tr_idx, y_all[tr_idx], groups_all[tr_idx]))
        idx_train = tr_idx[tr2_idx]; idx_dev = tr_idx[dev_rel]; idx_test = te_idx

    # проверки на утечки
    Gtr = set(groups_all[idx_train]); Gdv = set(groups_all[idx_dev]); Gte = set(groups_all[idx_test])
    assert not (Gtr & Gte), f"LEAK train↔test по origin: {sorted(list(Gtr & Gte))[:5]} ..."
    assert not (Gdv & Gte), f"LEAK dev↔test по origin: {sorted(list(Gdv & Gte))[:5]} ..."
    assert not (Gtr & Gdv), f"LEAK train↔dev по origin: {sorted(list(Gtr & Gdv))[:5]} ..."

    # в тесте не должно быть классов, которых нет в train
    train_classes = np.unique(y_all[idx_train])
    mask_te_keep = np.isin(y_all[idx_test], train_classes)
    if mask_te_keep.sum() < len(idx_test):
        idx_test = idx_test[mask_te_keep]

    def _uniq(arr): return len(set(arr))
    print(f"[split] items: train={len(idx_train)} dev={len(idx_dev)} test={len(idx_test)} | "
          f"origins: train={_uniq(groups_all[idx_train])} dev={_uniq(groups_all[idx_dev])} test={_uniq(groups_all[idx_test])}")

    ensure_dir(args.out_dir)
    pd.DataFrame({
        "path":   [paths[i] for i in np.concatenate([idx_train, idx_dev, idx_test])],
        "origin": [groups_all[i] for i in np.concatenate([idx_train, idx_dev, idx_test])],
        "label":  [int(y_all[i]) for i in np.concatenate([idx_train, idx_dev, idx_test])],
        "split":  (["train"]*len(idx_train) + ["dev"]*len(idx_dev) + ["test"]*len(idx_test))
    }).to_csv(os.path.join(args.out_dir, "split_groups.csv"), index=False)

    # Норм-статы
    ensure_dir(args.out_dir)
    mean, std = compute_norm_stats([ (paths[i], y_all[i]) for i in idx_train ], feat_dim, args.downsample, max_len)
    std = np.maximum(std, 1e-2).astype(np.float32)
    np.savez_compressed(os.path.join(args.out_dir, "norm_stats.npz"), mean=mean, std=std, max_len=max_len)

    # TF / стратегия
    tf, layers, models = lazy_tf()
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass

    if args.mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")

    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None
    replicas = strategy.num_replicas_in_sync if strategy else 1
    global_bs = args.batch_size * replicas
    print(f"GPUs: {len(gpus)} | replicas: {replicas} | global_batch: {global_bs}")

    # Датасеты
    make_ds = make_datasets(paths, y_all, max_len, feat_dim, global_bs,
                            args.downsample, mean, std, replicas,
                            shuffle_buf=1024, prefetch_n=2, num_parallel_calls=2, clip_abs=8.0)
    train_ds = make_ds(idx_train, shuffle=True,  drop_remainder=(replicas > 1))
    dev_ds   = make_ds(idx_dev,   shuffle=False, drop_remainder=False)
    test_ds  = make_ds(idx_test,  shuffle=False, drop_remainder=False)

    # классы и веса
    classes_in_train = np.unique(y_all[idx_train])
    num_class = int(classes_in_train.size)
    assert classes_in_train.min()==0 and classes_in_train.max()==num_class-1, \
        f"Train labels must be 0..K-1, got {classes_in_train}"

    from sklearn.utils.class_weight import compute_class_weight
    cls = np.arange(num_class, dtype=np.int32)
    w = compute_class_weight(class_weight="balanced", classes=cls, y=y_all[idx_train])
    class_weight = {int(c): float(wi) for c, wi in zip(cls, w)}
    print("class_weight:", class_weight)

    # Модель
    ctx = strategy.scope() if strategy else contextlib_null()
    with ctx:
        model = build_lstm_multiclass((max_len, feat_dim), num_class=num_class, learning_rate=1e-3, mixed_precision=args.mixed_precision)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    hist = model.fit(
        train_ds, validation_data=dev_ds,
        epochs=args.epochs, class_weight=class_weight,
        callbacks=callbacks, verbose=1
    )

    # предсказания и метрики
    prob_dev  = model.predict(dev_ds,  verbose=0).astype(np.float32)  # [N,K]
    y_dev     = y_all[idx_dev]
    pred_dev  = prob_dev.argmax(axis=1)

    prob_test = model.predict(test_ds, verbose=0).astype(np.float32)
    y_test    = y_all[idx_test]
    pred_test = prob_test.argmax(axis=1)

    acc = accuracy_score(y_test, pred_test)
    f1_macro = f1_score(y_test, pred_test, average="macro")
    f1_micro = f1_score(y_test, pred_test, average="micro")
    print("\n=== TEST ===")
    # красивые имена классов: возьмём из raw2zero
    zero2raw = {z:r for r,z in raw2zero.items()}
    class_names = [str(zero2raw.get(i, i)) for i in range(num_class)]
    print(classification_report(y_test, pred_test, target_names=class_names, digits=3))

    # Плоты и артефакты
    save_history_plots(hist, args.out_dir)
    cm = confusion_matrix(y_test, pred_test)
    save_confusion_matrix(cm, class_names, os.path.join(args.out_dir, "cm_test.png"))
  
    save_roc_ovr(y_test, prob_test, class_names, os.path.join(args.out_dir, "roc_ovr_test.png"))
    

    # сохранения
    model.save(os.path.join(args.out_dir, "model.keras"))
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"raw_to_zero_based": {str(k): int(v) for k,v in raw2zero.items()},
                   "classes_in_train": list(map(int, classes_in_train)),
                   "class_names": class_names},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": float(acc), "f1_macro": float(f1_macro), "f1_micro": float(f1_micro)}, f, indent=2)

    print("Saved to:", args.out_dir)

if __name__ == "__main__":
    main()
