---
title: "ITimeSeriesModel<T>"
description: "Defines the core functionality for time series prediction models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the core functionality for time series prediction models.

## How It Works

Time series models analyze sequential data points collected over time to identify patterns
and make predictions about future values.

**For Beginners:** A time series model helps you predict future values based on past data that
was collected in sequence over time. For example:

- Predicting tomorrow's temperature based on weather patterns from the past week
- Forecasting next month's sales based on previous months' sales data
- Estimating website traffic for next week based on historical visitor counts

These models look for patterns in your historical data (like trends, seasonal effects, and cycles)
to make educated guesses about what will happen next.

This interface inherits from IModelSerializer, which means these models can be saved to disk
and loaded back later - useful for when you've trained a good model and want to use it again
without retraining.

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance using test data. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the provided input vector. |

