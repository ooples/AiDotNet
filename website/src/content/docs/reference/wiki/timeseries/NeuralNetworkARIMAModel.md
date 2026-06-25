---
title: "NeuralNetworkARIMAModel<T>"
description: "Represents a Neural Network ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Represents a Neural Network ARIMA (Autoregressive Integrated Moving Average) model for time series forecasting.

## For Beginners

This model is like a super-powered crystal ball for predicting future values in a sequence of data.

Imagine you're trying to predict tomorrow's temperature:

- The ARIMA part looks at recent temperatures and how they've been changing.
- The Neural Network part can spot complex patterns, like how weekends or holidays might affect temperature.

By combining these two approaches, this model can make more accurate predictions than either method alone.
It's especially useful for data that changes over time, like stock prices, weather patterns, or sales figures.

## How It Works

This class combines traditional ARIMA modeling with neural networks to create a hybrid model for time series forecasting.
It incorporates both linear (ARIMA) and non-linear (neural network) components to capture complex patterns in the data.

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
    .ConfigureModel(new NeuralNetworkARIMAModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"NeuralNetworkARIMAModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetworkARIMAModel(NeuralNetworkARIMAOptions<>)` | Initializes a new instance of the `NeuralNetworkARIMAModel` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOptimized` | Gets whether parameter optimization succeeded during the most recent training run. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAverageNNContribution` | Computes an average neural network contribution for JIT approximation. |
| `ComputeResiduals(Matrix<>,Vector<>)` | Computes the residuals (errors) of the model predictions. |
| `CreateDefaultNeuralNetwork` | Creates a default neural network architecture for the model. |
| `CreateInstance` | Creates a new instance of the Neural Network ARIMA model with the same configuration. |
| `CreateNeuralNetworkInput(Vector<>,Vector<>,Int32)` | Creates the input vector for the neural network component of the model. |
| `DeserializeCore(BinaryReader)` | Deserializes the core components of the model from a binary reader. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the model's performance using various metrics. |
| `GetModelMetadata` | Gets metadata about the model, including type, parameters, and configuration. |
| `InitializeParameters` | Initializes the AR and MA parameters of the model with small random values. |
| `OptimizeParameters(Matrix<>,Vector<>)` | Optimizes the model parameters using the provided input data and target values. |
| `Predict(Matrix<>)` | Predicts values for the given input data. |
| `PredictSingle(Vector<>)` | Predicts a single value for the given input vector. |
| `PredictSingle(Vector<>,Vector<>,Int32)` | Predicts a single value based on previous predictions and current input. |
| `SerializeCore(BinaryWriter)` | Serializes the core components of the model to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training process for the Neural Network ARIMA model. |
| `UpdateModelParameters(Vector<>)` | Updates the model parameters with the optimized values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_arParameters` | Coefficients for the Autoregressive (AR) component of the model. |
| `_fitted` | The predicted values for the training data. |
| `_maParameters` | Coefficients for the Moving Average (MA) component of the model. |
| `_neuralNetwork` | The neural network component of the model. |
| `_nnarimaOptions` | Configuration options for the Neural Network ARIMA model. |
| `_optimizer` | The optimization algorithm used to find the best parameter values. |
| `_residuals` | The prediction errors (residuals) for each training example. |
| `_y` | The target values used during training. |

