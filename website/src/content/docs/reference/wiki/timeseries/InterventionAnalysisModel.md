---
title: "InterventionAnalysisModel<T>"
description: "Represents a model that analyzes and forecasts time series data with interventions or structural changes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Represents a model that analyzes and forecasts time series data with interventions or structural changes.

## For Beginners

Intervention analysis helps understand how specific events affect your data patterns.

Think of it like analyzing sales data:

- You've been tracking monthly sales that follow a regular pattern
- Then you run a major marketing campaign in July
- Sales jump significantly and stay higher for several months

This model helps you:

- Measure exactly how much the marketing campaign boosted sales
- Understand how long the effect lasted
- Make better predictions by accounting for these special events

Other examples of interventions include policy changes, natural disasters, product launches,
or any significant event that changes the normal pattern of your data.

## How It Works

Intervention analysis combines ARIMA (AutoRegressive Integrated Moving Average) modeling with the ability to 
account for external events or interventions that cause structural changes in the time series. These interventions 
can be temporary or permanent and can have various effects on the level, trend, or seasonality of the data.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;

double[] series =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};
var x = new Matrix<double>(series.Length, 1);
for (int i = 0; i < series.Length; i++) x[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new InterventionAnalysisModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"InterventionAnalysisModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InterventionAnalysisModel(InterventionAnalysisOptions<,Matrix<>,Vector<>>)` | Initializes a new instance of the `InterventionAnalysisModel` class with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeResiduals(Matrix<>,Vector<>)` | Computes the residuals between the actual and predicted values. |
| `CreateInstance` | Creates a new instance of the intervention analysis model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance on test data. |
| `GetInterventionEffects` | Gets a dictionary of the estimated intervention effects. |
| `GetModelMetadata` | Returns metadata about the model, including its type, parameters, and configuration. |
| `InitializeParameters` | Initializes the model parameters with small random values. |
| `OptimizeParameters(Matrix<>,Vector<>)` | Optimizes the model parameters to minimize prediction error. |
| `Predict(Matrix<>)` | Generates predictions for the given input data. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input vector. |
| `PredictSingle(Vector<>,Int32)` | Predicts a single value at the specified index. |
| `Reset` | Resets the model to its initial state. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the intervention analysis model. |
| `UpdateModelParameters(Vector<>)` | Updates the model parameters with the optimized values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_arParameters` | The autoregressive (AR) parameters of the model. |
| `_fitted` | The fitted (predicted) values for the training data. |
| `_iaOptions` | The configuration options for the intervention analysis model. |
| `_interventionEffects` | The effects of interventions on the time series. |
| `_maParameters` | The moving average (MA) parameters of the model. |
| `_optimizer` | The optimizer used to find the best model parameters. |
| `_residuals` | The residuals (errors) of the model predictions on the training data. |
| `_y` | The target values (observed time series data) used for training. |

