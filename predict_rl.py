#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


def run_feats(data_dir: Path, schema: Path, out_csv: Path):
    cmd = [
        "python", "detect_inj/feats.py",
        "--data_dir", str(data_dir),
        "--out_csv",  str(out_csv),
        "--schema",   str(schema),
    ]
    subprocess.run(cmd, check=True)


def prepare_data_dir(inp: Path) -> Path:
    """Если на входе файл — копируем его во временную папку.
       Если папка — используем её как есть."""
    if inp.is_dir():
        return inp
    if inp.suffix.lower() != ".npy":
        raise SystemExit("Ожидаю .npy файл или директорию с .npy")
    tmpdir = Path(tempfile.mkdtemp(prefix="npy_infer_"))
    shutil.copy2(inp, tmpdir / inp.name)
    return tmpdir


def main():
    ap = argparse.ArgumentParser(description="Predict on npy: feats -> XGB model")
    ap.add_argument("--input", required=True, help="Путь к .npy файлу или директории с .npy")
    ap.add_argument("--schema", required=True, help="schema_joints.json для feats.py")
    ap.add_argument("--model_dir", required=True, help="Папка с xgb.json и features_cols.json (например /kaggle/working/out1)")
    ap.add_argument("--out_csv", default="/kaggle/working/preds.csv")
    ap.add_argument("--thr", type=float, default=0.5, help="Порог для метки (по умолчанию 0.5)")
    args = ap.parse_args()

    inp = Path(args.input)
    data_dir = prepare_data_dir(inp)

    # 1) посчитать фичи
    tmp_feats_dir = Path(tempfile.mkdtemp(prefix="feats_"))
    feats_csv = tmp_feats_dir / "features.csv"
    run_feats(data_dir, Path(args.schema), feats_csv)

    # 2) загрузить фичи и привести колонки под модель
    F = pd.read_csv(feats_csv)
    fname_col = next((c for c in ("stem", "basename", "file", "filename") if c in F.columns), None)
    if fname_col is None:
        raise SystemExit("В features.csv нет колонки с именем файла (stem/basename/file/filename)")

    cols_needed = json.load(open(Path(args.model_dir) / "features_cols.json"))
    X_df = pd.DataFrame({c: (F[c] if c in F.columns else 0.0) for c in cols_needed}, dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)

    # 3) загрузить модель и предсказать
    clf = xgb.XGBClassifier()
    clf.load_model(str(Path(args.model_dir) / "xgb.json"))
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= args.thr).astype(int)

    # 4) сохранить результат
    out = pd.DataFrame({
        "filename": F[fname_col].astype(str).map(lambda s: Path(s).name),
        "prob_injury": prob,
        "pred": pred,            # 1 = injury, 0 = no injury
    })
    out.to_csv(args.out_csv, index=False)
    print(out)
    print(f"\nSaved predictions -> {args.out_csv}")


if __name__ == "__main__":
    main()
