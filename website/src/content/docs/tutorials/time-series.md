---
title: "Time Series"
description: "Forecasting and anomaly detection."
order: 8
section: "Tutorials"
---

Learn how to forecast future values from temporal data using AiDotNet.

## Overview

The most portable way to forecast through the `AiModelBuilder` facade is **lagged-window regression**: turn a series into `(window → next value)` training pairs and fit any regressor. This works with every model AiDotNet ships and predicts through the standard `result.Predict(...)`.

AiDotNet also includes dedicated statistical and deep time-series models (`ARIMAModel`, `SARIMAModel`, `ProphetModel`, `ExponentialSmoothing`, `DeepARModel`) for when you need their specific forecasting behavior.

---

## Quick Start: Lagged-Window Forecasting

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Historical monthly sales.
double[] sales =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};

// Frame the series into 3-month windows -> next month.
const int window = 3;
var rows = new List<double[]>();
var targets = new List<double>();
for (int i = 0; i + window < sales.Length; i++)
{
    rows.Add(sales.Skip(i).Take(window).ToArray());
    targets.Add(sales[i + window]);
}

var X = ToMatrix(rows.ToArray());
var y = new Vector<double>(targets.ToArray());

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GradientBoostingRegression<double>(
        new GradientBoostingRegressionOptions { NumberOfTrees = 100 }))
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(X, y))
    .BuildAsync();

// Forecast the next month from the most recent window.
var lastWindow = new Matrix<double>(1, window);
for (int j = 0; j < window; j++) lastWindow[0, j] = sales[^(window - j)];
Console.WriteLine($"Next month forecast: {result.Predict(lastWindow)[0]:F0}");

// Backtest accuracy off the result.
var stats = result.GetDataSetStats(X, y);
Console.WriteLine($"R²: {stats.PredictionStats.R2:F4}, RMSE: {stats.ErrorStats.RMSE:F2}");

static Matrix<double> ToMatrix(double[][] r)
{
    var m = new Matrix<double>(r.Length, r[0].Length);
    for (int i = 0; i < r.Length; i++)
        for (int j = 0; j < r[0].Length; j++)
            m[i, j] = r[i][j];
    return m;
}
```

---

## Available Time-Series Models

| Model | Description |
|:------|:------------|
| `ARIMAModel` | AutoRegressive Integrated Moving Average (univariate) |
| `SARIMAModel` | Seasonal ARIMA |
| `ExponentialSmoothing` | Holt-Winters trend + seasonality |
| `ProphetModel` | Business-style forecasting with holidays/seasonality |
| `DeepARModel` | Autoregressive RNN for probabilistic forecasting |

These implement `ITimeSeriesModel<T>` (an `IFullModel`), so they configure through `ConfigureModel(new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 1 }))`. They expose their own forecasting APIs for multi-step horizons.

---

## Feature Engineering

Lag and rolling-window features sharpen any forecaster. AiDotNet ships transformers for both:

| Transformer | Produces |
|:------------|:---------|
| `LagLeadTransformer` | Lagged (and lead) values at chosen offsets |
| `RollingStatsTransformer` | Rolling mean / std / min / max over a window |

Build a richer feature matrix with these before training, then feed it to any regressor exactly as in the quick-start above.

---

## Best Practices

1. **Check stationarity**: difference the series (or use `ARIMAModel`'s `D`) before modeling trends.
2. **Handle seasonality**: use `SARIMAModel` or add seasonal lag features.
3. **Scale your data**: neural models need normalized input (`ConfigurePreprocessing(...)`).
4. **Walk-forward validation**: evaluate on later windows, never a random split.
5. **Watch the horizon**: accuracy degrades the further ahead you forecast.

---

## Next Steps

- [Regression Tutorial](/docs/tutorials/regression/) — for non-temporal prediction
- [Deployment Tutorial](/docs/tutorials/deployment/) — serve your models in production
