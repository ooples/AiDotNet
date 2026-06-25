---
title: "MeanBiasErrorMetric<T>"
description: "Computes Mean Bias Error (MBE): average signed error showing systematic over/under-prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Mean Bias Error (MBE): average signed error showing systematic over/under-prediction.

## For Beginners

Mean Bias Error tells you if your model systematically over or under-predicts:

- MBE = 0: No systematic bias (over and under-predictions cancel out)
- MBE > 0: Model tends to over-predict (predictions are too high on average)
- MBE < 0: Model tends to under-predict (predictions are too low on average)

Note: MBE alone doesn't tell you about accuracy - use alongside MAE or RMSE.
A model could have MBE ≈ 0 but terrible accuracy if errors are random but balanced.

## How It Works

MBE = (1/N) * Σ(ŷ_i - y_i)

