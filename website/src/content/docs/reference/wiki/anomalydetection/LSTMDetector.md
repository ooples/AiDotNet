---
title: "LSTMDetector<T>"
description: "Implements LSTM-based anomaly detection using prediction error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Implements LSTM-based anomaly detection using prediction error.

## For Beginners

LSTM (Long Short-Term Memory) learns patterns in sequential data
and predicts the next value. Anomalies are detected when the actual value differs
significantly from the predicted value.

## How It Works

The algorithm follows the prediction-based formulation of Malhotra et al. (2015) and
Hundman et al. (2018, NASA telemanom):

1. Train an LSTM to FORECAST the next value of the series.
2. For each point, compute the forecast error.
3. High forecast errors indicate anomalies.

Because the core of the detector IS a forecasting LSTM, the model also exposes a normal
forecasting surface: `Train(Matrix<T>, Vector<T>)` learns a target series and
`Predict` returns one-step-ahead forecasts. The recurrent core runs on the
shared tensor Engine (tape-based BPTT) rather than a hand-rolled scalar loop.

Reference: Malhotra et al. (2015), "Long Short Term Memory Networks for Anomaly Detection
in Time Series"; Hundman et al. (2018), "Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding"; Hochreiter & Schmidhuber (1997), "Long Short-Term Memory".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTMDetector(Int32,Int32,Int32,Double,Double,Int32)` | Creates a new LSTM anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDim` | Gets the hidden dimension of the LSTM. |
| `SeqLength` | Gets the sequence/window length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `Forecast(Int32)` | Returns one-step-ahead forecasts of the learned series in the ORIGINAL (denormalized) scale — the model's underlying prediction surface (Malhotra et al. |
| `ForecastInSample(Int32)` | One-step-ahead in-sample forecasts (denormalized), feature 0, padded/continued to `steps` positions. |
| `PredictWindows` | Runs every length-_seqLength window of the stored series through the forecaster and returns the last-position forecast of each window: `windowForecast[w]` is the (normalized) prediction of series position `w + _seqLength`. |
| `ScoreAnomalies(Matrix<>)` |  |

