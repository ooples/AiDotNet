---
title: "Time Series"
description: "Forecasting and anomaly detection."
order: 8
section: "Tutorials"
---

Learn how to forecast future values from temporal data using AiDotNet.

## Overview

Dedicated time-series models — `ARIMAModel`, `SARIMAModel`, `ProphetModel`, `ExponentialSmoothing`, `DeepARModel` — forecast straight through the facade's unified `result.Predict(...)`: **the number of rows you ask for is the forecast horizon**, and the forecast extends the series the model was trained on. There's no separate `Forecast` call — one `Predict` front for every model.

You can also forecast with **lagged-window regression**: turn a series into `(window → next value)` pairs and fit any regressor — handy when you want a tree/boosting model.

---

## Quick Start: Forecasting with ARIMA

Train an `ARIMAModel` on your series, then ask `result.Predict` for an N-row matrix to get an N-step-ahead forecast.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;

// Monthly sales history.
double[] sales =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};

// ARIMA forecasts from the series itself; X is a placeholder sized to the series.
var n = sales.Length;
var seriesX = new Matrix<double>(n, 1);
for (int i = 0; i < n; i++) seriesX[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 1 }))
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(seriesX, new Vector<double>(sales)))
    .BuildAsync();

// Ask for 6 rows -> a 6-step-ahead forecast through the unified Predict.
var forecast = result.Predict(new Matrix<double>(6, 1));
for (int i = 0; i < forecast.Length; i++)
    Console.WriteLine($"Month +{i + 1}: {forecast[i]:F0}");
```

---

## Alternative: Lagged-Window Regression

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

These implement `ITimeSeriesModel<T>` (an `IFullModel`), so they configure through `ConfigureModel(new ARIMAModel<double>(...))` and forecast through the same `result.Predict(N-row matrix)` shown above — the row count is the horizon.

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
