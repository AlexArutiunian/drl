#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, shutil, subprocess, tempfile, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

# ---------------- utils ----------------
def run_feats(data_dir: Path, schema: Path | None, out_csv: Path,
              filelist_csv: Path, filelist_col: str = "filename"):
    """Считает фичи только по файлам из filelist_csv через drl/feats.py.
       Если schema=None — ключ --schema не передаём (feats.py возьмёт дефолт j0..)."""
    cmd = [
        "python", "drl/feats.py",
        "--data_dir", str(data_dir),
        "--out_csv",  str(out_csv),
        "--filelist_csv", str(filelist_csv),
        "--filelist_col", filelist_col,
    ]
    if schema is not None:
        cmd += ["--schema", str(schema)]
    print(f"[feats] run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def prepare_data_dir(inp: Path) -> Path:
    if inp.is_dir():
        return inp
    if inp.suffix.lower() != ".npy":
        raise SystemExit("Ожидаю .npy файл или директорию с .npy")
    tmpdir = Path(tempfile.mkdtemp(prefix="npy_infer_"))
    shutil.copy2(inp, tmpdir / inp.name)
    return tmpdir

def stem_lower(s: str) -> str:
    b = os.path.basename(str(s))
    return os.path.splitext(b)[0].lower()

def basename_lower(s: str) -> str:
    return os.path.basename(str(s)).lower()

# ----- plots -----
def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_roc_ovr(y_true_zero, prob, class_names, out_path, title="ROC (OVR)"):
    K = prob.shape[1]
    y_bin = label_binarize(y_true_zero, classes=list(range(K)))
    fpr, tpr, roc_auc = {}, {}, {}
    valid = []
    for i in range(K):
        yi = y_bin[:, i]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        fi, ti, _ = roc_curve(yi, prob[:, i])
        fpr[i], tpr[i] = fi, ti
        roc_auc[i] = roc_auc_score(yi, prob[:, i])
        valid.append(i)
    if not valid:
        return
    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid)
    macro_auc = roc_auc_score(y_bin[:, valid], prob[:, valid], average="macro", multi_class="ovr")
    plt.figure(figsize=(6,5), dpi=150)
    for i in valid:
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.9, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(all_fpr, mean_tpr, lw=2.0, label=f"macro-average (AUC={macro_auc:.3f})")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def save_confidence_buckets_multiclass(prob, y_true_zero, y_pred_zero, out_path, model_name="xgb-mc"):
    prob = np.asarray(prob, dtype=np.float32)
    conf = prob.max(axis=1)
    bins = [0.0, 0.5, 0.6, 0.8, 1.0001]
    labels = ["conf <50%", "50–60%", "60–80%", ">80%"]
    def bucket_counts(mask):
        if mask.sum() == 0: return np.zeros(len(labels), dtype=int)
        h, _ = np.histogram(conf[mask], bins=bins); return h
    y_true_zero = np.asarray(y_true_zero, int); y_pred_zero = np.asarray(y_pred_zero, int)
    panels = [
        (np.ones_like(conf, dtype=bool),   f"{model_name} — ALL"),
        (y_pred_zero == y_true_zero,       f"{model_name} — CORRECT"),
        (y_pred_zero != y_true_zero,       f"{model_name} — WRONG"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=150)
    for ax, (mask, title) in zip(axs, panels):
        cnts = bucket_counts(mask)
        bars = ax.barh(range(len(labels)), cnts, height=0.55)
        ax.set_yticks(range(len(labels)), labels); ax.set_title(title, fontsize=10)
        ax.set_xlabel("Count"); xmax = max(int(cnts.max()), 1); ax.set_xlim(0, xmax * 1.15)
        ax.bar_label(bars, labels=[str(int(v)) for v in cnts], padding=3, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Predict on npy (multiclass-ready): feats -> XGB model; predict_csv only; nice logs")
    ap.add_argument("--input", required=True, help="Путь к .npy файлу или директории с .npy")
    ap.add_argument("--schema", default=None, help="schema_joints.json для feats.py. Если не указан: возьму из --model_dir/schema_joints.json, иначе feats.py использует дефолт j0..")
    ap.add_argument("--model_dir", required=True, help="Папка с xgb_multiclass.json и features_cols.json (+ label_mapping.json)")
    ap.add_argument("--out_csv", default="preds.csv", help="Куда сохранить предсказания")
    # CSV со списком файлов и (опц.) метками
    ap.add_argument("--predict_csv", required=True, help="CSV со списком файлов для инференса; опционально с колонкой метки")
    ap.add_argument("--file_col", default="filename", help="Колонка с файлами в predict_csv (по умолчанию 'filename')")
    ap.add_argument("--label_col", default=None, help="Колонка с метками (raw; будет сопоставлена через label_mapping.json)")
    ap.add_argument("--encoding", default="utf-8-sig")
    # Куда сохранять фичи
    ap.add_argument("--feats_csv", default=None, help="Куда сохранить рассчитанные фичи (CSV). По умолчанию рядом с out_csv -> features_predict.csv")
    args = ap.parse_args()

    inp = Path(args.input); data_dir = prepare_data_dir(inp)
    model_dir = Path(args.model_dir)

    # ===== resolve schema
    schema_path: Path | None = Path(args.schema) if args.schema else None
    if schema_path is None:
        cand = model_dir / "schema_joints.json"
        if cand.exists():
            schema_path = cand
            print(f"[schema] Using schema from model_dir: {cand}")
        else:
            print("[schema] No --schema and no schema_joints.json in model_dir; feats.py will use default joint names (j0..).")

    # ===== read predict_csv
    PRED = pd.read_csv(args.predict_csv, encoding=args.encoding)
    if args.file_col not in PRED.columns:
        alt = next((c for c in PRED.columns if c.strip().lower() == args.file_col.strip().lower()), None)
        if alt is None:
            raise SystemExit(f"[err] В {args.predict_csv} нет колонки '{args.file_col}'. Найдено: {list(PRED.columns)}")
        args.file_col = alt
    print(f"[read] predict_csv: {args.predict_csv}  rows={len(PRED)}  cols={len(PRED.columns)}  file_col='{args.file_col}'")
    print("[head] predict_csv (first 5 rows):"); print(PRED.head(5).to_string(index=False))

    all_npy = sorted([p for p in Path(data_dir).rglob("*.npy")])
    print(f"[scan] input dir: {data_dir}  npy files found: {len(all_npy)}")
    for p in all_npy[:5]: print("   -", p.name)

    # keys
    PRED["_key_base"] = PRED[args.file_col].astype(str).map(basename_lower)
    PRED["_key_stem"] = PRED[args.file_col].astype(str).map(stem_lower)
    print(f"[keys] unique stems in predict_csv: {PRED['_key_stem'].nunique()}  bases: {PRED['_key_base'].nunique()}")

    # ===== features out path
    feats_csv = Path(args.feats_csv) if args.feats_csv else (Path(args.out_csv).parent if Path(args.out_csv).parent.as_posix() != "" else Path(".")) / "features_predict.csv"
    feats_csv.parent.mkdir(parents=True, exist_ok=True)

    # ===== run feats ONLY for files in predict_csv
    run_feats(data_dir=data_dir, schema=schema_path, out_csv=feats_csv, filelist_csv=Path(args.predict_csv), filelist_col=args.file_col)

    # ===== load features
    F = pd.read_csv(feats_csv)
    print(f"[feats] loaded: {feats_csv}  rows={len(F)}  cols={len(F.columns)}")
    fname_col = next((c for c in ("stem", "basename", "file", "filename") if c in F.columns), None)
    if fname_col is None:
        raise SystemExit("В features.csv нет колонки с именем файла (stem/basename/file/filename)")
    F["_key_base"] = F[fname_col].astype(str).map(basename_lower)
    F["_key_stem"] = F[fname_col].astype(str).map(stem_lower)
    print("[head] features (first 5 rows):")
    cols_show = [c for c in [fname_col, "_key_base", "_key_stem", "n_frames", "n_joints"] if c in F.columns]
    print(F[cols_show].head(5).to_string(index=False))

    # filter by stems intersection (robust)
    stems_csv = set(PRED["_key_stem"]); stems_feat = set(F["_key_stem"]); inter = stems_csv & stems_feat
    print(f"[match] stems: csv={len(stems_csv)}  feats={len(stems_feat)}  intersection={len(inter)}")
    if inter: print("[match] example stems:", list(sorted(inter))[:5])
    F = F.merge(PRED[["_key_stem"]].drop_duplicates(), on="_key_stem", how="inner")
    if F.empty:
        raise SystemExit("[err] Нет пересечения между файлами в predict_csv и фичами, посчитанными в data_dir.")

    # ===== align columns
    cols_needed = json.load(open(model_dir / "features_cols.json"))
    X_df = pd.DataFrame({c: (F[c] if c in F.columns else 0.0) for c in cols_needed}, dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)

    # ===== load model (multiclass)
    model_path = model_dir / "xgb_multiclass.json"
    if not model_path.exists():
        print("[warn] xgb_multiclass.json не найден — пытаюсь xgb.json")
        model_path = model_dir / "xgb.json"
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))

    proba = clf.predict_proba(X)  # [N,K]
    y_pred_zero = proba.argmax(axis=1)
    y_pred_topprob = proba.max(axis=1)

    # ===== label mapping (zero <-> raw)
    label_map_path = model_dir / "label_mapping.json"
    if label_map_path.exists():
        lm = json.load(open(label_map_path, "r"))
        raw_to_zero = {int(k): int(v) for k, v in lm.get("label_to_zero_based", {}).items()}
        zero_to_raw = {v: k for k, v in raw_to_zero.items()}
        classes_in_train = sorted(map(int, lm.get("classes_in_train", list(zero_to_raw.keys()))))
    else:
        print("[warn] label_mapping.json не найден — считаю, что raw == zero")
        zero_to_raw = {i: i for i in range(proba.shape[1])}
        classes_in_train = list(range(proba.shape[1]))

    # human-readable file name back from predict_csv
    rev_name = (PRED[[args.file_col, "_key_stem"]]
                .drop_duplicates("_key_stem")
                .set_index("_key_stem"))[args.file_col].to_dict()

    # ===== build predictions dataframe
    out = pd.DataFrame({
        "_key_stem": F["_key_stem"],
        "filename": F["_key_stem"].map(lambda k: rev_name.get(k, k)),
        "y_pred_zero": y_pred_zero.astype(int),
        "y_pred": [zero_to_raw.get(int(z), int(z)) for z in y_pred_zero],
        "y_pred_topprob": y_pred_topprob.astype(float)
    })
    # per-class probabilities with raw labels in column names
    for j in range(proba.shape[1]):
        raw_lbl = zero_to_raw.get(int(j), int(j))
        out[f"proba_{raw_lbl}"] = proba[:, j]

    out_csv = Path(args.out_csv); out.to_csv(out_csv, index=False)
    print("\n[head] predictions (first 5 rows):"); print(out.head(5).to_string(index=False))
    print(f"\nSaved predictions -> {out_csv}")
    print(f"Features saved -> {feats_csv}")

    # ===== metrics if labels present (multiclass)
    label_col = args.label_col
    if label_col is None:
        for cand in ["label", "target", "y"]:
            if cand in PRED.columns: label_col = cand; break

    if label_col and label_col in PRED.columns:
        Y = PRED[[label_col, "_key_stem"]].copy()
        Y["label_raw"] = pd.to_numeric(Y[label_col], errors="coerce")
        Y = Y.dropna(subset=["label_raw"])
        Y["label_raw"] = Y["label_raw"].astype(int)

        y_true_merge = out.merge(Y[["_key_stem", "label_raw"]], on="_key_stem", how="inner")
        if y_true_merge.empty:
            print("[warn] Метки не сопоставились с предсказаниями — метрики пропущены.")
            return

        if label_map_path.exists():
            y_true_zero = y_true_merge["label_raw"].map(raw_to_zero).astype("int32")
        else:
            if y_true_merge["label_raw"].min() == 0 and y_true_merge["label_raw"].max() == proba.shape[1]-1:
                y_true_zero = y_true_merge["label_raw"].astype("int32")
            else:
                y_true_zero = (y_true_merge["label_raw"].astype(int) - 1).clip(lower=0).astype("int32")

        y_pred_zero_eval = y_true_merge["y_pred_zero"].astype("int32").to_numpy()
        y_true_zero_eval = y_true_zero.to_numpy()
        prob_eval = proba[y_true_merge.index.values, :]

        acc = accuracy_score(y_true_zero_eval, y_pred_zero_eval)
        f1_macro = f1_score(y_true_zero_eval, y_pred_zero_eval, average="macro")
        f1_micro = f1_score(y_true_zero_eval, y_pred_zero_eval, average="micro")
        try:
            auc_macro_ovr = roc_auc_score(y_true_zero_eval, prob_eval, multi_class="ovr", average="macro")
        except Exception:
            auc_macro_ovr = np.nan

        print("\n[metrics-mc]")
        print("Accuracy:", round(float(acc), 4))
        print("F1-macro:", round(float(f1_macro), 4))
        print("F1-micro:", round(float(f1_micro), 4))
        print("Macro AUC (OVR):", "nan" if np.isnan(auc_macro_ovr) else round(float(auc_macro_ovr), 4))
        cm = confusion_matrix(y_true_zero_eval, y_pred_zero_eval)
        print("Confusion matrix (rows=true, cols=pred):\n", cm)

        K = proba.shape[1]
        class_names = [str(zero_to_raw.get(i, i)) for i in range(K)]
        print("\nReport:\n", classification_report(y_true_zero_eval, y_pred_zero_eval, target_names=class_names, digits=3))

        out_dir = out_csv.parent if out_csv.parent.as_posix() != "" else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_true_zero_eval, y_pred_zero_eval, target_names=class_names, digits=4))
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv", index=True)

        metrics_payload = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "auc_macro_ovr": None if np.isnan(auc_macro_ovr) else float(auc_macro_ovr),
            "n_eval": int(len(y_true_zero_eval)),
            "classes": class_names,
        }
        with open(out_dir / "metrics_test.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        save_confusion_matrix(cm, class_names, out_dir / "cm.png", title="Confusion Matrix (predict_csv, multiclass)")
        save_roc_ovr(y_true_zero_eval, prob_eval, class_names, out_dir / "roc_ovr.png", title="ROC OVR (predict_csv)")
        save_confidence_buckets_multiclass(prob_eval, y_true_zero_eval, y_pred_zero_eval, out_dir / "confidence_buckets.png", model_name="xgb-mc")
    else:
        print("[info] В predict_csv нет колонки с метками — метрики не посчитаны.")

if __name__ == "__main__":
    main()
