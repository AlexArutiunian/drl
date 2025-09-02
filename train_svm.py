#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_svm.py — классическая SVM (SVC) для бинарной и мультиклассовой классификации
по .npy-последовательностям. Без утечек: групповой сплит по origin (из CSV или из имени).

Пример:
python train_svm.py \
  --csv /kaggle/working/aug_multi/data_split.csv \
  --data_dir /kaggle/working/aug_multi/npy_all \
  --filename_col filename \
  --label_col label \
  --out_dir output_svm_multi \
  --downsample 3 \
  --kernel rbf --C 10 --gamma scale
"""

from __future__ import annotations
import os, re, json, math, argparse
from typing import List, Tuple, Dict, Optional
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# sklearn
from sklearn.model_selection import GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)

from sklearn.utils.class_weight import compute_class_weight
import joblib

# plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============== utils ===============

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
        if x and x not in cands: cands.append(x)
    push(os.path.join(data_dir, rel))
    if not rel.endswith(".npy"): push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json"):
        base = rel[:-5]; push(os.path.join(data_dir, base + ".npy")); push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json.npy"): push(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))
    b = os.path.basename(rel)
    push(os.path.join(data_dir, b))
    if not b.endswith(".npy"): push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json"):
        push(os.path.join(data_dir, b[:-5] + ".npy")); push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json.npy"): push(os.path.join(data_dir, b.replace(".json.npy",".npy")))
    return cands

def pick_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p): return p
    return None

# универсальная обработка сырых меток -> ключ (int если можно, иначе строка)
def to_key(v):
    s = str(v).strip().lower()
    try:
        f = float(s)
        if math.isfinite(f): return int(round(f))
    except Exception:
        pass
    return s

# агрегатные фичи из (T,F)
def _features_from_seq(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.shape[0] < 2:
        return np.zeros(x.shape[1]*6, dtype=np.float32)
    dif = np.diff(x, axis=0)
    stat = np.concatenate([
        np.nanmean(x, axis=0), np.nanstd(x, axis=0),
        np.nanmin(x, axis=0),  np.nanmax(x, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ]).astype(np.float32, copy=False)
    return np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

def extract_features(path: str, downsample: int = 1) -> np.ndarray:
    arr = np.load(path, allow_pickle=False, mmap_mode="r")
    x = sanitize_seq(arr)
    if downsample > 1: x = x[::downsample]
    return _features_from_seq(x)

# ========= графики / отчёты =========

def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)\
        .plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def save_roc_binary(y_true, prob, out_path, title="ROC (binary)"):
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5), dpi=150)
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], "--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path); plt.close()
    return roc_auc

def save_roc_ovr(y_true: np.ndarray, prob: np.ndarray, class_names: List[str], out_path: str):
    # one-vs-rest (OVR)
    K = prob.shape[1]
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

def save_confidence_buckets(prob: np.ndarray, y_true: np.ndarray, out_dir: str,
                            class_names: List[str], model_name="svm", thr: float | None = None):
    """
    Для binary: prob.shape=[N] (вероятность класса 1), confidence = max(p,1-p)
    Для multi:  prob.shape=[N,K], confidence = max_k prob[n,k]
    Рисуем три панели: ALL / CORRECT / WRONG.
    """
    prob = np.asarray(prob)
    if prob.ndim == 1:          # binary
        pred = (prob >= (0.5 if thr is None else float(thr))).astype(int)
        conf = np.maximum(prob, 1.0 - prob)
    else:                       # multiclass
        pred = prob.argmax(axis=1)
        conf = prob.max(axis=1)

    y_true = np.asarray(y_true).astype(int)
    correct = (pred == y_true)

    bins = [0.0, 0.5, 0.6, 0.8, 1.0000001]
    labels = ["conf <50%", "50–60%", "60–80%", ">80%"]

    def bucket_counts(mask):
        h, _ = np.histogram(conf[mask], bins=bins)
        return h

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=150)
    panels = [
        (np.ones_like(conf, dtype=bool), f"{model_name} — ALL", axs[0]),
        (correct,                         f"{model_name} — CORRECT", axs[1]),
        (~correct,                        f"{model_name} — WRONG", axs[2]),
    ]

    for mask, title, ax in panels:
        cnts = bucket_counts(mask)
        bars = ax.barh(range(len(labels)), cnts, height=0.55)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Count")
        xmax = max(1, int(cnts.max()))
        ax.set_xlim(0, xmax * 1.25)
        for i, b in enumerate(bars):
            v = int(cnts[i])
            ax.text(v + 0.02 * xmax, b.get_y() + b.get_height() / 2, str(v),
                    va="center", fontsize=9, clip_on=False)
        ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_buckets.png"), dpi=150)
    plt.close(fig)


# =============== индекс и группы (anti-leak) ===============

ORX = re.compile(r'(?:_chunk\d+|_aug\d+|_\d+)$', re.IGNORECASE)

def origin_from_name(path_or_name: str) -> str:
    st = os.path.splitext(os.path.basename(str(path_or_name)))[0].lower()
    while True:
        new = ORX.sub('', st)
        if new == st: break
        st = new
    return st

def build_items(csv_path: str, data_dir: str,
                filename_col="filename", label_col="label",
                debug_index: bool=False):
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


# =============== main ===============

def main():
    ap = argparse.ArgumentParser("SVM (SVC) over features from NPY sequences — binary or multiclass, with group split")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", default="output_svm")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--peek", type=int, default=0)
    ap.add_argument("--debug_index", action="store_true")

    # SVM hyperparams
    ap.add_argument("--kernel", choices=["rbf","linear","poly","sigmoid"], default="rbf")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", default="scale")  # 'scale' | 'auto' | float
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--coef0", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--max_iter", type=int, default=-1)

    # threshold tuning (binary only)
    ap.add_argument("--threshold_mode", default="bal_acc",
                    choices=["bal_acc", "f1_pos", "macro_f1", "roc_j", "spec", "fixed"])
    ap.add_argument("--threshold_fixed", type=float, default=None)
    ap.add_argument("--target_specificity", type=float, default=None)

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Индекс
    paths, y_raw, stats, _ = build_items(
        args.csv, args.data_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        debug_index=args.debug_index
    )
    assert paths and y_raw, "Не найдено валидных .npy и меток"

    # Маппинг сырых меток -> 0..K-1
    raw_vals = [to_key(v) for v in y_raw]
    uniq_raw = list(dict.fromkeys(raw_vals))
    raw2zero = {rv:i for i, rv in enumerate(sorted(uniq_raw, key=lambda x: (isinstance(x,str), x)))}
    y_all = np.array([raw2zero[to_key(v)] for v in y_raw], dtype=np.int32)
    zero2raw = {z:r for r,z in raw2zero.items()}
    class_names = [str(zero2raw.get(i, i)) for i in range(len(raw2zero))]
    num_class = int(len(class_names))

    # Группы (anti-leak)
    DF_SRC = pd.read_csv(args.csv)
    name_col = args.filename_col if args.filename_col in DF_SRC.columns else \
        next((c for c in DF_SRC.columns if c.lower().strip()==args.filename_col.lower()), None)
    origin_col = next((c for c in DF_SRC.columns if c.lower()=="origin"), None)

    csv_origin_map: Dict[str,str] = {}
    if name_col is not None and origin_col is not None:
        for n, o in zip(DF_SRC[name_col], DF_SRC[origin_col]):
            if pd.isna(n): continue
            csv_origin_map[os.path.basename(str(n)).lower()] = str(o).lower()

    groups_all = []
    for p in paths:
        key = os.path.basename(p).lower()
        g = csv_origin_map.get(key) or origin_from_name(key)
        groups_all.append(g)
    groups_all = np.array(groups_all)

    # Сплит (StratifiedGroupKFold или GroupShuffleSplit)
    idx_all = np.arange(len(paths))
    if HAS_SGKF and num_class > 1:
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

    print(f"[split] items: train={len(idx_train)} dev={len(idx_dev)} test={len(idx_test)} | "
          f"origins: train={len(Gtr)} dev={len(Gdv)} test={len(Gte)}")

    # Извлечём фичи (можно заранее на все, это безопасно; скейлер — только по train)
    def batch_feats(indices):
        feats = []
        for i in tqdm(indices, desc="feats", leave=False):
            feats.append(extract_features(paths[i], downsample=args.downsample))
        return np.stack(feats).astype(np.float32, copy=False)

    X_train = batch_feats(idx_train)
    X_dev   = batch_feats(idx_dev)
    X_test  = batch_feats(idx_test)
    y_train, y_dev, y_test = y_all[idx_train], y_all[idx_dev], y_all[idx_test]

    # Стандартизация по train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev   = scaler.transform(X_dev)
    X_test  = scaler.transform(X_test)

    # Веса классов (на всякий случай — SVC с balanced сам это делает, но пригодится)
    classes = np.unique(y_train)
    try:
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(wi) for c, wi in zip(classes, w)}
    except Exception:
        class_weight = "balanced"
    print("class_weight:", class_weight)

    # --- Модель SVM ---
    gamma_val = args.gamma
    if gamma_val not in ("scale","auto"):
        try: gamma_val = float(gamma_val)
        except Exception: gamma_val = "scale"

    clf = SVC(
        kernel=args.kernel, C=args.C, gamma=gamma_val, degree=args.degree,
        coef0=args.coef0, tol=args.tol, max_iter=args.max_iter,
        probability=True, class_weight=class_weight, random_state=42
    )
    clf.fit(X_train, y_train)

    # ---- Оценки ----
    ensure_dir(args.out_dir)
    model_art = {"model":"SVC", "kernel":args.kernel, "C":args.C, "gamma":str(args.gamma),
                 "degree":args.degree, "coef0":args.coef0}

    if num_class == 2:
        # --- Binary: tune threshold on DEV ---
        prob_dev  = clf.predict_proba(X_dev)[:, 1]
        # threshold search
        def best_threshold(y_true, p, mode="bal_acc", fixed=None, target_spec=None):
            from sklearn.metrics import precision_recall_curve, balanced_accuracy_score
            p = np.asarray(p, dtype=np.float64)
            p = np.clip(p, 1e-7, 1-1e-7)
            if fixed is not None: return float(fixed)
            if mode == "f1_pos":
                pr, rc, th = precision_recall_curve(y_true, p)
                if len(th)==0: return 0.5
                f1 = 2*pr[:-1]*rc[:-1]/np.clip(pr[:-1]+rc[:-1], 1e-12, None)
                return float(th[int(np.nanargmax(f1))])
            if mode == "macro_f1":
                thr = np.linspace(0.01, 0.99, 199)
                scores = [f1_score(y_true, p>=t, average="macro") for t in thr]
                return float(thr[int(np.argmax(scores))])
            if mode == "bal_acc":
                thr = np.linspace(0.01, 0.99, 199)
                scores = [balanced_accuracy_score(y_true, p>=t) for t in thr]
                bi = int(np.argmax(scores))
                print(f"[THR] bal_acc={scores[bi]:.3f} @ thr={thr[bi]:.3f}")
                return float(thr[bi])
            if mode in ("roc_j","spec"):
                fpr, tpr, th = roc_curve(y_true, p)
                m = ~np.isinf(th); fpr, tpr, th = fpr[m], tpr[m], th[m]
                spec = 1.0 - fpr
                if mode != "spec":
                    j = tpr - fpr
                    return float(th[int(np.argmax(j))])
                target_spec = 0.5 if target_spec is None else float(target_spec)
                idx = np.where(spec >= target_spec)[0]
                return float(th[idx[-1]] if len(idx) else th[-1])
            return 0.5

        thr = best_threshold(y_dev, prob_dev, mode=args.threshold_mode,
                             fixed=args.threshold_fixed, target_spec=args.target_specificity)

        prob_test = clf.predict_proba(X_test)[:, 1]
        pred_dev  = (prob_dev  >= thr).astype(int)
        pred_test = (prob_test >= thr).astype(int)

        acc  = accuracy_score(y_test, pred_test)
        f1m  = f1_score(y_test, pred_test)
        aucv = roc_auc_score(y_test, prob_test)
        cm   = confusion_matrix(y_test, pred_test)

        dev_report  = classification_report(y_dev,  pred_dev,  target_names=class_names, digits=3)
        test_report = classification_report(y_test, pred_test, target_names=class_names, digits=3)

        # save metrics jsons
        with open(os.path.join(args.out_dir, "metrics_dev.json"), "w", encoding="utf-8") as f:
            json.dump({"accuracy": float(accuracy_score(y_dev, pred_dev)),
                       "f1": float(f1_score(y_dev, pred_dev)),
                       "roc_auc": float(roc_auc_score(y_dev, prob_dev)),
                       "confusion_matrix": confusion_matrix(y_dev, pred_dev).tolist(),
                       "report": dev_report},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
            json.dump({"accuracy": float(acc),
                       "f1": float(f1m),
                       "roc_auc": float(aucv),
                       "confusion_matrix": cm.tolist(),
                       "report": test_report},
                      f, ensure_ascii=False, indent=2)

        # plots
        save_confusion_matrix(cm, class_names, os.path.join(args.out_dir, "cm_test.png"))
        save_roc_binary(y_test, prob_test, os.path.join(args.out_dir, "roc_test.png"))
        save_confidence_buckets(prob_test, y_test, args.out_dir, class_names, model_name="svm", thr=thr)

        # also DEV visuals
        save_confusion_matrix(confusion_matrix(y_dev, pred_dev), class_names, os.path.join(args.out_dir, "cm_dev.png"))
        save_roc_binary(y_dev, prob_dev, os.path.join(args.out_dir, "roc_dev.png"))
        save_confidence_buckets(prob_dev, y_dev, args.out_dir, class_names, model_name="svm_dev", thr=thr)

        with open(os.path.join(args.out_dir, "threshold.txt"), "w") as f:
            f.write(str(thr))

        print("\n=== DEV (threshold tuned) ===")
        print(dev_report)
        print("\n=== TEST (using DEV threshold) ===")
        print(test_report)
        print(f"TEST: acc={acc:.3f} f1={f1m:.3f} roc_auc={aucv:.3f}")

    else:
        # --- Multiclass ---
        prob_dev  = clf.predict_proba(X_dev)    # [N,K]
        prob_test = clf.predict_proba(X_test)
        pred_dev  = prob_dev.argmax(axis=1)
        pred_test = prob_test.argmax(axis=1)

        acc  = accuracy_score(y_test, pred_test)
        f1ma = f1_score(y_test, pred_test, average="macro")
        f1mi = f1_score(y_test, pred_test, average="micro")
        cm   = confusion_matrix(y_test, pred_test)

        dev_report  = classification_report(y_dev,  pred_dev,  target_names=class_names, digits=3)
        test_report = classification_report(y_test, pred_test, target_names=class_names, digits=3)

        # JSON
        with open(os.path.join(args.out_dir, "metrics_dev.json"), "w", encoding="utf-8") as f:
            json.dump({"accuracy": float(accuracy_score(y_dev, pred_dev)),
                       "f1_macro": float(f1_score(y_dev, pred_dev, average="macro")),
                       "f1_micro": float(f1_score(y_dev, pred_dev, average="micro")),
                       "confusion_matrix": confusion_matrix(y_dev, pred_dev).tolist(),
                       "report": dev_report},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
            json.dump({"accuracy": float(acc),
                       "f1_macro": float(f1ma),
                       "f1_micro": float(f1mi),
                       "confusion_matrix": cm.tolist(),
                       "report": test_report},
                      f, ensure_ascii=False, indent=2)

        # plots
        save_confusion_matrix(cm, class_names, os.path.join(args.out_dir, "cm_test.png"))
        save_roc_ovr(y_test, prob_test, class_names, os.path.join(args.out_dir, "roc_ovr_test.png"))
        save_confidence_buckets(prob_test, y_test, args.out_dir, class_names, model_name="svm-multiclass")

        # DEV visuals
        save_confusion_matrix(confusion_matrix(y_dev, pred_dev), class_names, os.path.join(args.out_dir, "cm_dev.png"))
        save_roc_ovr(y_dev, prob_dev, class_names, os.path.join(args.out_dir, "roc_ovr_dev.png"))
        save_confidence_buckets(prob_dev, y_dev, args.out_dir, class_names, model_name="svm-multiclass-dev")

        print("\n=== DEV (multiclass) ===")
        print(dev_report)
        print("\n=== TEST (multiclass) ===")
        print(test_report)
        print(f"TEST: acc={acc:.3f} f1_macro={f1ma:.3f} f1_micro={f1mi:.3f}")

    # сохранения
    joblib.dump({"model": clf, "scaler": scaler, "label_mapping": raw2zero, "artifacts": model_art},
                os.path.join(args.out_dir, "svm_model.joblib"))
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"raw_to_zero_based": {str(k): int(v) for k,v in raw2zero.items()},
                   "class_names": class_names}, f, ensure_ascii=False, indent=2)

    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()
