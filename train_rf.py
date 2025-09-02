#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RandomForest trainer (binary & multiclass) with leakage-safe group split + confidence buckets.

Examples
--------
# NPY, бинарка или мультикласс (авто):
python rf_train.py \
  --csv /kaggle/working/aug/data_split.csv \
  --data_dir /kaggle/working/aug/npy_all \
  --filename_col filename \
  --label_col label \
  --out_dir outputs_rf \
  --downsample 2 \
  --n_estimators 600 --max_depth 20

# JSON (если нужно):
python rf_train.py \
  --csv meta.csv --data_dir json_dir --input_format json \
  --motion_key running --filename_col filename --label_col label
"""

import os, re, json, math, argparse
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

# viz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# save model
import joblib


# ===================== helpers =====================

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _nan_to_num32(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def sanitize_seq(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim >= 2:
        t_axis = int(np.argmax(a.shape))
        if t_axis != 0: a = np.moveaxis(a, t_axis, 0)
        if a.ndim > 2:  a = a.reshape(a.shape[0], -1)
    return a.astype(np.float32, copy=False)

def _features_from_seq(seq: np.ndarray) -> np.ndarray:
    """Агрегаты по времени для классики (устойчивые, без утечек)."""
    seq = seq.astype(np.float32, copy=False)
    dif = np.diff(seq, axis=0) if seq.shape[0] > 1 else np.zeros_like(seq)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
        np.nanmin(seq, axis=0),  np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0),
    ]).astype(np.float32, copy=False)
    return _nan_to_num32(stat)

def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    cands = []
    def push(x):
        if x and (x not in cands): cands.append(x)
    push(os.path.join(data_dir, rel))
    if not rel.endswith(".npy"): push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json"):
        base = rel[:-5]
        push(os.path.join(data_dir, base + ".npy"))
        push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json.npy"):
        push(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))
    b = os.path.basename(rel)
    push(os.path.join(data_dir, b))
    if not b.endswith(".npy"): push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json"):
        push(os.path.join(data_dir, b[:-5] + ".npy")); push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json.npy"):
        push(os.path.join(data_dir, b.replace(".json.npy",".npy")))
    return cands

def pick_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p): return p
    return None

# ----- label mapping (raw -> 0..K-1) -----
def _to_label_key(v):
    s = str(v).strip().lower()
    # частые бинарные варианты
    if s in ("no injury","no_injury","none","neg","negative"): return 0
    if s in ("injury","pos","positive"):                      return 1
    try:
        f = float(s)
        if math.isfinite(f): return int(round(f))
    except Exception:
        pass
    return s  # строка → отдельный класс

# ----- groups (origin) -----
ORX = re.compile(r'(?:_chunk\d+|_aug\d+|-\d+|_\d+)$', re.IGNORECASE)
def origin_from_name(path_or_name: str) -> str:
    st = os.path.splitext(os.path.basename(str(path_or_name)))[0].lower()
    while True:
        new = ORX.sub('', st)
        if new == st: break
        st = new
    return st


# ===================== JSON support (optional) =====================

def _safe_json_load(path):
    try:
        import orjson
        with open(path, "rb") as f: return orjson.loads(f.read())
    except Exception:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)

def _stack_motion_frames_with_schema(md: dict, schema_joints: List[str]) -> Optional[np.ndarray]:
    present = [j for j in schema_joints if j in md]
    if not present: return None
    T = min(len(md[j]) for j in present)
    if T <= 1: return None
    cols = []
    for j in schema_joints:
        if j in md:
            arr = np.asarray(md[j], dtype=np.float32)[:T]
        else:
            arr = np.full((T,3), np.nan, dtype=np.float32)
        cols.append(arr)
    return np.concatenate(cols, axis=1)


# ===================== load features =====================

def load_features_from_npy(csv_path, npy_dir, filename_col, label_col,
                           groups_col=None, downsample=1) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict]:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    lb_col = cols.get(label_col.lower(), label_col)
    gp_col = (cols.get(groups_col.lower()) if groups_col else None)

    paths, raw_labels, groups = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Index NPY", dynamic_ncols=True, mininterval=0.2):
        fn = str(row.get(fn_col, "")).strip()
        if not fn: continue
        path = pick_existing_path(possible_npy_paths(npy_dir, fn))
        if not path: continue
        y_raw = row.get(lb_col, None)
        key = _to_label_key(y_raw)
        if key is None: continue
        # group
        g = None
        if gp_col and (gp_col in df.columns):
            g = str(row.get(gp_col, "")).strip().lower()
        if not g: g = origin_from_name(path)
        # load -> features
        try:
            arr = np.load(path, allow_pickle=False, mmap_mode="r")
            x = sanitize_seq(arr)
            if downsample > 1: x = x[::downsample]
            if x.shape[0] < 2: continue
            feat = _features_from_seq(x)
            paths.append(path); raw_labels.append(key); groups.append(g)
            if 'D' not in locals(): D = feat.size
            if 'F' not in locals(): F = []
        except Exception:
            continue

    assert paths, "Нет валидных NPY"
    # map labels → 0..K-1
    uniq = list(dict.fromkeys(raw_labels))
    uniq_sorted = sorted(uniq, key=lambda z: (isinstance(z,str), z))
    raw2zero = {rv:i for i, rv in enumerate(uniq_sorted)}
    y = np.array([raw2zero[v] for v in raw_labels], dtype=np.int32)

    # features (вторым проходом — чтобы не хранить все seq)
    X = []
    for p in tqdm(paths, desc="Feats NPY", dynamic_ncols=True, mininterval=0.2):
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)
        if downsample > 1: x = x[::downsample]
        X.append(_features_from_seq(x))
    X = np.stack(X).astype(np.float32, copy=False)

    return X, y, paths, groups, {"raw2zero": raw2zero, "class_names": [str(s) for s in uniq_sorted]}

def load_features_from_json(csv_path, data_dir, filename_col, label_col,
                            motion_key="running", schema_joints=None,
                            groups_col=None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict]:
    assert schema_joints, "Для JSON нужен список суставов (schema_joints)"
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    lb_col = cols.get(label_col.lower(), label_col)
    gp_col = (cols.get(groups_col.lower()) if groups_col else None)

    paths, raw_labels, groups, feats = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Index JSON", dynamic_ncols=True, mininterval=0.2):
        fn = str(row.get(fn_col, "")).strip()
        if not fn: continue
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p) and not fn.endswith(".json"):
            p2 = p + ".json"
            p = p2 if os.path.exists(p2) else p
        if not os.path.exists(p): continue

        y_raw = row.get(lb_col, None)
        key = _to_label_key(y_raw)
        if key is None: continue

        g = None
        if gp_col and (gp_col in df.columns):
            g = str(row.get(gp_col, "")).strip().lower()
        if not g: g = origin_from_name(p)

        try:
            data = _safe_json_load(p)
            if motion_key not in data or not isinstance(data[motion_key], dict): continue
            seq = _stack_motion_frames_with_schema(data[motion_key], schema_joints)
            if seq is None or seq.shape[0] < 2: continue
            feats.append(_features_from_seq(seq))
            paths.append(p); raw_labels.append(key); groups.append(g)
        except Exception:
            continue

    assert feats, "Нет валидных JSON"
    uniq = list(dict.fromkeys(raw_labels))
    uniq_sorted = sorted(uniq, key=lambda z: (isinstance(z,str), z))
    raw2zero = {rv:i for i, rv in enumerate(uniq_sorted)}
    y = np.array([raw2zero[v] for v in raw_labels], dtype=np.int32)
    X = np.stack(feats).astype(np.float32, copy=False)
    return X, y, paths, groups, {"raw2zero": raw2zero, "class_names": [str(s) for s in uniq_sorted]}


# ===================== split (group-safe) =====================

def group_stratified_split(y: np.ndarray, groups: List[str], seed=42):
    """70/10/20 без пересечений по groups. Ставит dev = 12.5% от train."""
    idx_all = np.arange(len(y))
    groups = np.asarray(groups)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)   # ~20% test
        tr_idx, te_idx = next(sgkf.split(idx_all, y, groups))
        sgkf2 = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed+1) # ~12.5% dev of train
        tr_rel, dv_rel = next(sgkf2.split(tr_idx, y[tr_idx], groups[tr_idx]))
        idx_tr = tr_idx[tr_rel]; idx_dv = tr_idx[dv_rel]; idx_te = te_idx
    except Exception:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr_idx, te_idx = next(gss.split(idx_all, y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125/0.80, random_state=seed+1)
        tr2, dv_rel = next(gss2.split(tr_idx, y[tr_idx], groups[tr_idx]))
        idx_tr = tr_idx[tr2]; idx_dv = tr_idx[dv_rel]; idx_te = te_idx

    # safety checks
    Gtr, Gdv, Gte = set(groups[idx_tr]), set(groups[idx_dv]), set(groups[idx_te])
    assert not (Gtr & Gte), "LEAK train↔test"
    assert not (Gdv & Gte), "LEAK dev↔test"
    assert not (Gtr & Gdv), "LEAK train↔dev"
    # drop test samples with unseen classes
    seen = set(np.unique(y[idx_tr]))
    keep = np.array([i for i in idx_te if y[i] in seen], dtype=int)
    return idx_tr, idx_dv, keep


# ===================== metrics / plots =====================

def best_threshold_by_f1(y_true, proba):
    pr, rc, th = precision_recall_curve(y_true, proba)
    if len(th) == 0: return 0.5
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    j = int(np.nanargmax(f1))
    return float(th[j])

def plot_confusion(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=class_names)\
        .plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title); fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_roc_binary(y_true, proba, out_path, title="ROC (binary)"):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5,4.5), dpi=150)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
    ax.plot([0,1],[0,1],"--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return auc_val

def plot_roc_ovr(y_true, proba, class_names, out_path, title="ROC (OVR)"):
    K = proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(K)))
    fpr, tpr, roc_auc, valid = {}, {}, {}, []
    for i in range(K):
        yi = y_bin[:, i]
        if yi.sum()==0 or yi.sum()==len(yi):  # no variance
            continue
        fi, ti, _ = roc_curve(yi, proba[:, i])
        fpr[i], tpr[i] = fi, ti
        roc_auc[i] = auc(fi, ti)
        valid.append(i)
    if not valid:
        return None
    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid)
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6.5,5), dpi=150)
    for i in valid:
        ax.plot(fpr[i], tpr[i], lw=1, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    ax.plot(all_fpr, mean_tpr, lw=2.0, label=f"macro (AUC={macro_auc:.3f})")
    ax.plot([0,1],[0,1],"--",lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return macro_auc

def save_confidence_buckets_binary(proba, out_dir, thr=0.5, model_name="rf"):
    proba = np.asarray(proba).ravel()
    pred = (proba >= thr).astype(int)
    conf = np.maximum(proba, 1.0 - proba)
    bins   = [0.0, 0.5, 0.6, 0.8, 1.0001]
    labels = ["conf <50%", "50–60%", "60–80%", ">80%"]

    def bar(ax, mask, title):
        cnts, _ = np.histogram(conf[mask], bins=bins)
        bars = ax.barh(range(len(labels)), cnts, height=0.55)
        ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Count"); ax.set_title(title, fontsize=10)
        xmax = max(1, int(cnts.max())); ax.set_xlim(0, xmax*1.2)
        for i,b in enumerate(bars):
            v = int(cnts[i]); ax.text(v + 0.02*xmax, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.2)
        return cnts

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=150)
    c_all  = bar(axs[0], np.ones_like(conf, bool), f"{model_name} — ALL")
    c_pos  = bar(axs[1], pred == 1,               f"{model_name} — predicted: 1")
    c_neg  = bar(axs[2], pred == 0,               f"{model_name} — predicted: 0")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confidence_buckets.png")); plt.close(fig)

    # CSV с подсчётами
    df = pd.DataFrame({
        "bucket": labels, "all": c_all, "pred_1": c_pos, "pred_0": c_neg
    })
    df.to_csv(os.path.join(out_dir, "confidence_buckets_counts.csv"), index=False)

def save_confidence_buckets_multiclass(proba, y_true, out_path_img, out_path_csv, model_name="rf"):
    proba = np.asarray(proba, dtype=np.float32)
    conf = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)
    bins = [0.0, 0.5, 0.6, 0.8, 1.0001]
    labels = ["conf <50%", "50–60%", "60–80%", ">80%"]

    def hist(mask):
        if mask.sum()==0: return np.zeros(len(labels), dtype=int)
        h, _ = np.histogram(conf[mask], bins=bins); return h

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=150)
    panels = [
        (np.ones_like(conf, bool), f"{model_name} — ALL", axs[0]),
        (y_pred == y_true,         f"{model_name} — CORRECT", axs[1]),
        (y_pred != y_true,         f"{model_name} — WRONG", axs[2]),
    ]
    rows = {}
    for mask, title, ax in panels:
        cnts = hist(mask); rows[title] = cnts
        bars = ax.barh(range(len(labels)), cnts, height=0.55)
        ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Count"); ax.set_title(title, fontsize=10)
        xmax = max(1, int(cnts.max())); ax.set_xlim(0, xmax*1.2)
        for i,b in enumerate(bars):
            v = int(cnts[i]); ax.text(v + 0.02*xmax, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path_img); plt.close(fig)

    df = pd.DataFrame({"bucket": labels,
                       "all": rows[f"{model_name} — ALL"],
                       "correct": rows[f"{model_name} — CORRECT"],
                       "wrong": rows[f"{model_name} — WRONG"]})
    df.to_csv(out_path_csv, index=False)


# ===================== main =====================

def main():
    ap = argparse.ArgumentParser("RandomForest (binary & multiclass) with group-safe split and confidence buckets")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--input_format", choices=["npy","json"], default="npy")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--groups_col", default="origin", help="колонка групп (если нет — выводим из имени файла)")

    ap.add_argument("--motion_key", default="running", help="для JSON")
    ap.add_argument("--schema_json", default="", help="json со списком суставов (для JSON). Если пусто — не требуется.")

    ap.add_argument("--downsample", type=int, default=1, help="шаг по времени для NPY")
    ap.add_argument("--out_dir", default="output_rf")
    ap.add_argument("--seed", type=int, default=42)

    # RF hyperparams
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--max_features", default="auto")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # ==== load features ====
    if args.input_format == "npy":
        X_all, y_all, paths, groups, mapping = load_features_from_npy(
            args.csv, args.data_dir, args.filename_col, args.label_col,
            groups_col=args.groups_col, downsample=max(1, args.downsample)
        )
    else:
        schema = None
        if args.schema_json and os.path.exists(args.schema_json):
            with open(args.schema_json, "r", encoding="utf-8") as f:
                schema = json.load(f)
        assert schema, "Для JSON укажи --schema_json со списком суставов"
        X_all, y_all, paths, groups, mapping = load_features_from_json(
            args.csv, args.data_dir, args.filename_col, args.label_col,
            motion_key=args.motion_key, schema_joints=schema, groups_col=args.groups_col
        )

    class_names = mapping["class_names"]
    num_class = len(class_names)
    print(f"Classes: {num_class} -> {class_names}")

    # ==== split (group-safe) ====
    idx_tr, idx_dv, idx_te = group_stratified_split(y_all, groups, seed=args.seed)

    # save split map
    pd.DataFrame({
        "path":   [paths[i] for i in np.concatenate([idx_tr, idx_dv, idx_te])],
        "group":  [groups[i] for i in np.concatenate([idx_tr, idx_dv, idx_te])],
        "label":  [int(y_all[i]) for i in np.concatenate([idx_tr, idx_dv, idx_te])],
        "split":  (["train"]*len(idx_tr) + ["dev"]*len(idx_dv) + ["test"]*len(idx_te))
    }).to_csv(os.path.join(args.out_dir, "split_groups.csv"), index=False)

    # ==== scale (fit on train only) ====
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[idx_tr])
    X_dv = scaler.transform(X_all[idx_dv])
    X_te = scaler.transform(X_all[idx_te])
    y_tr, y_dv, y_te = y_all[idx_tr], y_all[idx_dv], y_all[idx_te]

    # ==== class weights (balanced) ====
    cls = np.arange(num_class, dtype=int)
    cw = compute_class_weight(class_weight="balanced", classes=cls, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(cls, cw)}
    print("class_weight:", class_weight)

    # ==== RF model ====
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-1, class_weight="balanced", random_state=args.seed
    )
    rf.fit(X_tr, y_tr)

    # ==== predictions ====
    if num_class == 2:
        prob_dv  = rf.predict_proba(X_dv)[:,1]
        prob_te  = rf.predict_proba(X_te)[:,1]
        thr = best_threshold_by_f1(y_dv, prob_dv)
        pred_dv  = (prob_dv >= thr).astype(int)
        pred_te  = (prob_te >= thr).astype(int)

        dev_metrics = {
            "threshold": float(thr),
            "accuracy": float(accuracy_score(y_dv, pred_dv)),
            "f1": float(f1_score(y_dv, pred_dv)),
            "roc_auc": float(roc_auc_score(y_dv, prob_dv)) if len(np.unique(y_dv))==2 else float("nan"),
            "confusion_matrix": confusion_matrix(y_dv, pred_dv).tolist(),
            "report": classification_report(y_dv, pred_dv, digits=3),
        }
        test_metrics = {
            "threshold": float(thr),
            "accuracy": float(accuracy_score(y_te, pred_te)),
            "f1": float(f1_score(y_te, pred_te)),
            "roc_auc": float(roc_auc_score(y_te, prob_te)) if len(np.unique(y_te))==2 else float("nan"),
            "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
            "report": classification_report(y_te, pred_te, digits=3),
        }

        # plots
        plot_confusion(y_dv, pred_dv,  [f"class_{i}" for i in range(num_class)], os.path.join(args.out_dir, "cm_dev.png"))
        plot_confusion(y_te, pred_te,  [f"class_{i}" for i in range(num_class)], os.path.join(args.out_dir, "cm_test.png"))
        plot_roc_binary(y_dv, prob_dv, os.path.join(args.out_dir, "roc_dev.png"))
        plot_roc_binary(y_te, prob_te, os.path.join(args.out_dir, "roc_test.png"))
        # PR (bonus)
        pr, rc, _ = precision_recall_curve(y_te, prob_te)
        ap = average_precision_score(y_te, prob_te)
        plt.figure(figsize=(5.5,4.5), dpi=150)
        plt.plot(rc, pr, label=f"AP={ap:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("PR (binary)"); plt.legend(); plt.grid(alpha=0.25)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "pr_test.png")); plt.close()

        # confidence buckets
        save_confidence_buckets_binary(prob_te, args.out_dir, thr=thr, model_name="rf")

    else:
        prob_dv = rf.predict_proba(X_dv)      # [N,K]
        prob_te = rf.predict_proba(X_te)      # [N,K]
        pred_dv = prob_dv.argmax(axis=1)
        pred_te = prob_te.argmax(axis=1)

        dev_metrics = {
            "accuracy": float(accuracy_score(y_dv, pred_dv)),
            "f1_macro": float(f1_score(y_dv, pred_dv, average="macro")),
            "f1_micro": float(f1_score(y_dv, pred_dv, average="micro")),
            "confusion_matrix": confusion_matrix(y_dv, pred_dv).tolist(),
            "report": classification_report(y_dv, pred_dv, target_names=[str(c) for c in class_names], digits=3),
        }
        test_metrics = {
            "accuracy": float(accuracy_score(y_te, pred_te)),
            "f1_macro": float(f1_score(y_te, pred_te, average="macro")),
            "f1_micro": float(f1_score(y_te, pred_te, average="micro")),
            "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
            "report": classification_report(y_te, pred_te, target_names=[str(c) for c in class_names], digits=3),
        }

        # plots
        plot_confusion(y_dv, pred_dv, class_names, os.path.join(args.out_dir, "cm_dev.png"))
        plot_confusion(y_te, pred_te, class_names, os.path.join(args.out_dir, "cm_test.png"))
        plot_roc_ovr(y_dv, prob_dv, class_names, os.path.join(args.out_dir, "roc_ovr_dev.png"))
        plot_roc_ovr(y_te, prob_te, class_names, os.path.join(args.out_dir, "roc_ovr_test.png"))
        save_confidence_buckets_multiclass(prob_te, y_te,
                                           os.path.join(args.out_dir, "confidence_buckets.png"),
                                           os.path.join(args.out_dir, "confidence_buckets_counts.csv"),
                                           model_name="rf")

    # ==== save artifacts ====
    joblib.dump({"model": rf, "scaler": scaler, "label_mapping": mapping},
                os.path.join(args.out_dir, "rf_model.joblib"))
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "metrics_dev.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # text summary
    print("\n=== DEV ===")
    for k,v in dev_metrics.items():
        if k!="confusion_matrix" and k!="report": print(f"{k}: {v}")
    print(dev_metrics["report"])
    print("\n=== TEST ===")
    for k,v in test_metrics.items():
        if k!="confusion_matrix" and k!="report": print(f"{k}: {v}")
    print(test_metrics["report"])
    print("\nSaved to:", args.out_dir)


if __name__ == "__main__":
    main()
