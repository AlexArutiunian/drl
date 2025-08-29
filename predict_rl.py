#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, shutil, subprocess, tempfile, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# --- plotting (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------- utils ----------------
def run_feats(data_dir: Path, schema: Path, out_csv: Path,
              filelist_csv: Path | None = None, filelist_col: str = "filename"):
    """
    Вызывает drl/feats.py — считает фичи только по файлам из filelist_csv.
    feats.py должен поддерживать --filelist_csv и --filelist_col.
    """
    cmd = [
        "python", "drl/feats.py",
        "--data_dir", str(data_dir),
        "--out_csv",  str(out_csv),
        "--schema",   str(schema),
        "--filelist_csv", str(filelist_csv),
        "--filelist_col", filelist_col,
    ]
    print(f"[feats] run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def prepare_data_dir(inp: Path) -> Path:
    """Если на входе файл — копируем его во временную папку. Если папка — используем её как есть."""
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

def parse_label_binary(v) -> int | None:
    """Нормализует метку к {0,1} или возвращает None, если непонятно."""
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    s_clean = re.sub(r"[^a-z0-9]+", "", s)
    pos = {"1","yes","true","inj","injury","y"}
    neg = {"0","no","false","noinj","noinjury","n"}
    if s_clean in pos: return 1
    if s_clean in neg: return 0
    try:
        f = float(s)
        if f == 1.0: return 1
        if f == 0.0: return 0
    except Exception:
        pass
    if "inj" in s and "no" not in s: return 1
    if "no" in s and "inj" in s: return 0
    return None

# ----- plots -----
def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_roc_binary(y_true, prob_pos, out_path, title="ROC (binary)"):
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc = roc_auc_score(y_true, prob_pos)
    plt.figure(figsize=(6,5), dpi=150)
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_confidence_buckets_binary(prob_pos, y_true, y_pred, out_path, model_name="xgb-binary"):
    prob_pos = np.asarray(prob_pos, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    conf = np.maximum(prob_pos, 1.0 - prob_pos)  # уверенность top-1
    bins = [0.0, 0.5, 0.6, 0.8, 1.0001]
    labels = ["conf <50%", "50–60%", "60–80%", ">80%"]

    def bucket_counts(mask):
        if mask.sum() == 0:
            return np.zeros(len(labels), dtype=int)
        h, _ = np.histogram(conf[mask], bins=bins)
        return h

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
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Predict on npy: feats -> XGB model (predict_csv only, nice logs & feats path)")
    ap.add_argument("--input", required=True, help="Путь к .npy файлу или директории с .npy")
    ap.add_argument("--schema", required=True, help="schema_joints.json для feats.py")
    ap.add_argument("--model_dir", required=True, help="Папка с xgb.json и features_cols.json")
    ap.add_argument("--out_csv", default="preds.csv", help="Куда сохранить предсказания")
    ap.add_argument("--thr", type=float, default=0.5, help="Порог для метки (по умолчанию 0.5)")

    # CSV со списком файлов и (опц.) метками
    ap.add_argument("--predict_csv", required=True, help="CSV со списком файлов для инференса; опционально с колонкой метки")
    ap.add_argument("--file_col", default="filename", help="Имя колонки с файлами в predict_csv (по умолчанию 'filename')")
    ap.add_argument("--label_col", default=None, help="Имя колонки с метками (0/1, injury/no inj). Если не задано, попробуем угадать.")
    ap.add_argument("--encoding", default="utf-8-sig")
    # КУДА сохранять фичи
    ap.add_argument("--feats_csv", default=None, help="Путь куда сохранить рассчитанные фичи (CSV). По умолчанию рядом с out_csv -> features_predict.csv")
    args = ap.parse_args()

    inp = Path(args.input)
    data_dir = prepare_data_dir(inp)

    # 0) загрузим список файлов и (опц.) метки из predict_csv
    PRED = pd.read_csv(args.predict_csv, encoding=args.encoding)
    if args.file_col not in PRED.columns:
        alt = next((c for c in PRED.columns if c.strip().lower() == args.file_col.strip().lower()), None)
        if alt is None:
            raise SystemExit(f"[err] В {args.predict_csv} нет колонки '{args.file_col}'. Найдено: {list(PRED.columns)}")
        args.file_col = alt

    # покажем, что читаем
    print(f"[read] predict_csv: {args.predict_csv}  rows={len(PRED)}  cols={len(PRED.columns)}  file_col='{args.file_col}'")
    print("[head] predict_csv (first 5 rows):")
    print(PRED.head(5).to_string(index=False))

    # просканируем input и покажем первые 5 файлов
    all_npy = sorted([p for p in Path(data_dir).rglob("*.npy")])
    print(f"[scan] input dir: {data_dir}  npy files found: {len(all_npy)}")
    for p in all_npy[:5]:
        print("   -", p.name)

    # нормализованные ключи для соединения по файлу
    PRED["_key_base"] = PRED[args.file_col].astype(str).map(basename_lower)
    PRED["_key_stem"] = PRED[args.file_col].astype(str).map(stem_lower)
    print(f"[keys] unique stems in predict_csv: {PRED['_key_stem'].nunique()}  bases: {PRED['_key_base'].nunique()}")

    # 1) посчитать фичи только по этим файлам — и сохранить в понятное место
    if args.feats_csv:
        feats_csv = Path(args.feats_csv)
    else:
        out_dir = Path(args.out_csv).parent if Path(args.out_csv).parent.as_posix() != "" else Path(".")
        feats_csv = out_dir / "features_predict.csv"
    feats_csv.parent.mkdir(parents=True, exist_ok=True)

    run_feats(
        data_dir=data_dir,
        schema=Path(args.schema),
        out_csv=feats_csv,
        filelist_csv=Path(args.predict_csv),
        filelist_col=args.file_col
    )

    # 2) загрузить фичи и привести колонки под модель
    F = pd.read_csv(feats_csv)
    print(f"[feats] loaded: {feats_csv}  rows={len(F)}  cols={len(F.columns)}")
    fname_col = next((c for c in ("stem", "basename", "file", "filename") if c in F.columns), None)
    if fname_col is None:
        raise SystemExit("В features.csv нет колонки с именем файла (stem/basename/file/filename)")

    # ключи для join
    F["_key_base"] = F[fname_col].astype(str).map(basename_lower)
    F["_key_stem"] = F[fname_col].astype(str).map(stem_lower)

    print("[head] features (first 5 rows):")
    cols_show = [c for c in [fname_col, "_key_base", "_key_stem", "n_frames", "n_joints"] if c in F.columns]
    print(F[cols_show].head(5).to_string(index=False))

    # пересечение: берём STRICT по stem (устойчиво к расширениям/пути)
    stems_csv = set(PRED["_key_stem"])
    stems_feat = set(F["_key_stem"])
    inter = stems_csv & stems_feat
    print(f"[match] stems: csv={len(stems_csv)}  feats={len(stems_feat)}  intersection={len(inter)}")
    if inter:
        ex = list(sorted(inter))[:5]
        print("[match] example stems:", ex)

    # фильтрация фичей: только файлы из predict_csv
    F = F.merge(PRED[["_key_stem"]].drop_duplicates(), on="_key_stem", how="inner")
    if F.empty:
        raise SystemExit("[err] Нет пересечения между файлами в predict_csv и фичами, посчитанными в data_dir.")

    # подгонка колонок под модель
    cols_needed = json.load(open(Path(args.model_dir) / "features_cols.json"))
    X_df = pd.DataFrame({c: (F[c] if c in F.columns else 0.0) for c in cols_needed}, dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)

    # 3) загрузить модель и предсказать
    clf = xgb.XGBClassifier()
    clf.load_model(str(Path(args.model_dir) / "xgb.json"))
    proba = clf.predict_proba(X)

    # поддержка бинарной и мультикласс — метрики ниже для бинарной
    if proba.shape[1] == 2:
        prob_pos = proba[:, 1]
        pred = (prob_pos >= args.thr).astype(int)
    else:
        pred = proba.argmax(axis=1)
        prob_pos = proba.max(axis=1)  # для confidence-плота

    # 4) сохранить предсказания
    rev = (PRED[[args.file_col, "_key_stem"]]
           .drop_duplicates("_key_stem")
           .set_index("_key_stem"))[args.file_col].to_dict()

    out = pd.DataFrame({
        "filename": F["_key_stem"].map(lambda k: rev.get(k, k)),
        "prob_injury": prob_pos,
        "pred": pred,
    })
    out_csv = Path(args.out_csv)
    out.to_csv(out_csv, index=False)
    print("\n[head] predictions (first 5 rows):")
    print(out.head(5).to_string(index=False))
    print(f"\nSaved predictions -> {out_csv}")
    print(f"Features saved -> {feats_csv}")

    # 5) метрики, если в predict_csv есть метки (бинарные)
    label_col = args.label_col
    if label_col is None:
        for cand in ["label", "target", "y", "injury", "No inj / inj", "No inj/inj", "noinj_inj"]:
            if cand in PRED.columns:
                label_col = cand
                break

    if label_col and label_col in PRED.columns and proba.shape[1] == 2:
        y_true_raw = PRED[[label_col, "_key_stem"]].copy()
        y_true_raw["y"] = y_true_raw[label_col].map(parse_label_binary)
        y_true_raw = y_true_raw.dropna(subset=["y"])
        y_true_raw["y"] = y_true_raw["y"].astype(int)

        M = out.merge(y_true_raw[["_key_stem","y"]], left_on=F["_key_stem"].name, right_on="_key_stem", how="inner")
        if len(M) == 0:
            print("[warn] Не удалось сопоставить метки с предсказаниями — пропускаю метрики.")
            return

        y_true = M["y"].to_numpy().astype(int)
        y_pred = M["pred"].to_numpy().astype(int)
        y_prob = M["prob_injury"].to_numpy().astype(float)

        acc = accuracy_score(y_true, y_pred)
        f1_bin = f1_score(y_true, y_pred, average="binary")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        auc_bin = roc_auc_score(y_true, y_prob)

        print("\n[metrics]")
        print("Accuracy:", round(float(acc), 4))
        print("F1 (binary pos=injury):", round(float(f1_bin), 4))
        print("F1-macro:", round(float(f1_macro), 4))
        print("F1-micro:", round(float(f1_micro), 4))
        print("ROC AUC:", round(float(auc_bin), 4))
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix (rows=true, cols=pred):\n", cm)
        print("\nReport:\n", classification_report(y_true, y_pred, target_names=["NoInj(0)","Inj(1)"], digits=3))

        out_dir = out_csv.parent if out_csv.parent.as_posix() != "" else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_true, y_pred, target_names=["NoInj(0)","Inj(1)"], digits=4))
        pd.DataFrame(cm, index=["NoInj(0)","Inj(1)"], columns=["NoInj(0)","Inj(1)"])\
          .to_csv(out_dir / "confusion_matrix.csv", index=True)

        metrics_payload = {
            "accuracy": float(acc),
            "f1_binary": float(f1_bin),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "roc_auc": float(auc_bin),
            "n_eval": int(len(y_true)),
        }
        with open(out_dir / "metrics_test.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        # картинки
        save_confusion_matrix(cm, ["NoInj(0)","Inj(1)"], out_dir / "cm.png", title="Confusion Matrix (predict_csv)")
        save_roc_binary(y_true, y_prob, out_dir / "roc.png", title="ROC (predict_csv)")
        save_confidence_buckets_binary(y_prob, y_true, y_pred, out_dir / "confidence_buckets.png", model_name="xgb-binary")
    else:
        if proba.shape[1] != 2:
            print("[info] Модель не бинарная (num_classes != 2) — метрики по predict_csv пропущены.")
        else:
            print("[info] В predict_csv нет колонки с метками (или не удалось определить) — метрики не посчитаны.")

if __name__ == "__main__":
    main()
