"""
Sanity-check telco_churn.onnx outside Databricks.

Reads the model + test CSV produced by Program.cs (run `dotnet run` first),
scores the CSV via Python onnxruntime, and confirms the predictions match
the `churn_predicted` column AiDotNet recorded — to within 1e-3.

Run:
    pip install onnxruntime numpy pandas
    python verify_locally.py
"""
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).parent
MODEL_PATH = HERE / "telco_churn.onnx"
CSV_PATH   = HERE / "telco_churn_test_data.csv"

for p in [MODEL_PATH, CSV_PATH]:
    if not p.exists():
        print(f"Missing {p}.  Run `dotnet run` in this folder first.", file=sys.stderr)
        sys.exit(2)
    print(f"OK  {p}  ({p.stat().st_size:,} bytes)")

sess = ort.InferenceSession(str(MODEL_PATH))
print(f"ONNX inputs:  {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
print(f"ONNX outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")

df = pd.read_csv(CSV_PATH)
print(f"\nLoaded {len(df):,} rows from {CSV_PATH.name}")

feature_cols = ["tenure_norm", "monthly_norm", "total_norm", "contract_norm"]
X = df[feature_cols].to_numpy(dtype=np.float32)
preds = sess.run(None, {"input": X})[0].flatten()
df["onnx_prediction"] = preds
df["abs_diff"] = (df["churn_predicted"] - df["onnx_prediction"]).abs()

print("\nFirst 5 rows:")
print(df.head(5)[["tenure_norm", "contract_norm", "churn_actual",
                  "churn_predicted", "onnx_prediction", "abs_diff"]].to_string(index=False))

print()
print(f"Max  abs diff vs AiDotNet:  {df['abs_diff'].max():.6f}")
print(f"Mean abs diff:              {df['abs_diff'].mean():.6f}")

over_tol = (df["abs_diff"] > 1e-3).sum()
if over_tol:
    print(f"\nFAIL: {over_tol} / {len(df)} rows differ from AiDotNet by more than 1e-3")
    sys.exit(1)

print(f"\nPASS: all {len(df)} rows match AiDotNet within 1e-3.")
print("Safe to upload telco_churn.onnx + telco_churn_test_data.csv to Databricks.")
