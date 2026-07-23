# Telco Churn ONNX Export Demo

Builds a small AiDotNet binary classifier and exports it to a `.onnx` file the Databricks training session uses for the `Databricks ↔ AiDotNet` integration demo.

## What this sample does

1. Generates 2,000 synthetic Telco-Churn-shaped rows (4 numeric features → binary churn target).
2. Builds a layer chain: `Dense(8) + ReLU → Dense(1) + Sigmoid`.
3. Warms the layers up with a forward pass so the lazy-init weights materialise.
4. Scores all 2,000 rows so we can capture expected outputs.
5. Exports the layer chain to `telco_churn.onnx` via the new `ConvertToOnnx` path (`feature/onnx-export`).
6. Writes `telco_churn_test_data.csv` with 200 sample rows (inputs + expected churn probability).

The sample deliberately **does not include a training step** in this revision. Adding `model.Train(...)` requires more setup against the current `NeuralNetwork`/`AiModelBuilder` API surface, and the Databricks demo's teaching point ("load an ONNX file in a notebook and score it against a Delta table") works equally well with random-init weights. A future revision can swap in real training once the AiModelBuilder neural-network training path is locked down.

## Running

```bash
dotnet run --project samples/deployment/TelcoChurnOnnxExport
```

Outputs (in the project's run directory):
- `telco_churn.onnx` — the model artifact for Databricks
- `telco_churn_test_data.csv` — sample inputs + expected outputs

## Verifying the .onnx file loads externally

Before handing the .onnx off to Databricks, sanity-check it with Python `onnxruntime`:

```bash
pip install onnxruntime numpy
python - <<'PY'
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("telco_churn.onnx")
print("Inputs: ", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("Outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

# Row 1 of the CSV: 0.3454, -0.7182, -0.4137, 0.0 → expected 0.4660
x = np.array([[0.3454, -0.7182, -0.4137, 0.0]], dtype=np.float32)
out = sess.run(None, {"input": x})
print("ONNX prediction:", out[0][0][0])
PY
```

Expected output: roughly `0.466` — matching the `churn_predicted` column in row 1 of the CSV within rounding (verified during development at ~1.6e-5 absolute difference).

## In the Databricks notebook

The notebook (built in a later phase of the training session) will:
1. Upload `telco_churn.onnx` to DBFS (or workspace files)
2. Install `onnxruntime` on the cluster
3. Load `telco_churn_test_data.csv` as a Delta table
4. Run inference with a `pandas_udf` (or Spark + `onnxruntime` Python API)
5. Compare predictions to the `churn_predicted` column to prove the AiDotNet model behaves identically inside Databricks

## Layer-by-layer ONNX coverage

Uses the new `LayerBase.ConvertToOnnx` virtual added on `feature/onnx-export`. See `docs/ONNX_SUPPORT_MATRIX.md` for the full list of supported layer types.
