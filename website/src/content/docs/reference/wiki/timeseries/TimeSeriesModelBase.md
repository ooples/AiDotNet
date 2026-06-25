---
title: "TimeSeriesModelBase<T>"
description: "Provides a base class for all time series forecasting models in the library."
section: "API Reference"
---

`Base Classes` · `AiDotNet.TimeSeries`

Provides a base class for all time series forecasting models in the library.

## For Beginners

A time series model helps predict future values based on past observations.

Think of a time series like a sequence of measurements taken over time - for example,
daily temperatures, monthly sales, or hourly website visits. These models analyze the patterns
in historical data to make predictions about what will happen next.

This base class is like a blueprint that all specific time series models follow.
It ensures that every model can:

- Be trained on historical data to learn patterns
- Make predictions for future periods based on what it learned
- Evaluate how accurate its predictions are compared to actual values
- Be saved to disk and loaded later without retraining

Time series models are used in many real-world applications, including:

- Weather forecasting
- Stock market prediction
- Demand planning for retail
- Energy consumption forecasting
- Website traffic prediction

## How It Works

This abstract class defines the common interface and functionality that all time series models share,
including training, prediction, evaluation, and serialization/deserialization capabilities.

Time series models capture temporal dependencies in data and use patterns learned from historical
observations to predict future values. This base class provides the foundation for implementing
various time series forecasting algorithms like ARIMA, Exponential Smoothing, TBATS, and more complex
machine learning approaches.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesModelBase(TimeSeriesRegressionOptions<>)` | Initializes a new instance of the TimeSeriesModelBase class with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vector operations. |
| `IsTrained` | Indicates whether the model has been trained. |
| `LastEvaluationMetrics` | Gets the last computed error metrics when the model was evaluated. |
| `ModelParameters` | Gets or sets the trained model parameters. |
| `NumOps` | Provides numeric operations for the specific type T. |
| `Options` | Configuration options for the time series model. |
| `SupportsParameterInitialization` | Gets the trainable parameters of the model as a vector. |
| `TrainingCancellationToken` | Trains the time series model using the provided input data and target values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyParameters(Vector<>)` | Applies the provided parameters to the model. |
| `BackpropagateLayers(Tensor<>)` | Backpropagates the loss gradient through the model's neural network layers. |
| `CalculateErrorMetrics(Vector<>,Vector<>)` | Calculates error metrics by comparing predictions to actual values. |
| `Clip(,,)` | Clips a value to be within the specified range. |
| `Clone` | Creates a clone of the time series model. |
| `CreateInstance` | Creates a new instance of the derived model class. |
| `DeepCopy` | Creates a deep copy of the time series model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeCore(BinaryReader)` | Deserializes model-specific data from the binary reader. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this time-series model. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the performance of the trained model on test data. |
| `Forecast(Vector<>,Int32)` | Generates a forecast for multiple steps ahead. |
| `GetActiveFeatureIndices` | Gets the indices of features (lags/time periods) actively used by the model. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets the feature importance scores as a dictionary. |
| `GetFeatureImportance(Int32)` | Gets the importance of a specific feature (lag). |
| `GetInputShape` |  |
| `GetLayerParameterGradients` | Extracts accumulated parameter gradients from all layers after backpropagation. |
| `GetModelMetadata` | Gets metadata about the time series model. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GuardPrediction(,Double)` | Validates the input data for prediction. |
| `IsFeatureUsed(Int32)` | Determines if a specific feature (lag) is actively used by the model. |
| `LoadState(Stream)` | Loads the time series model's state from a stream. |
| `Predict(Matrix<>)` | Generates forecasts using the trained time series model. |
| `PredictSingle(Vector<>)` | Generates a prediction for a single input vector. |
| `PrepareForecastFeatures(List<>,Int32)` | Prepares input features for a forecast step using the extended history. |
| `Reset` | Resets the model to its untrained state. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the time series model's current state to a stream. |
| `SerializeCore(BinaryWriter)` | Serializes model-specific data to the binary writer. |
| `SerializeForMetadata` | Serializes the model to a byte array for storage or transmission. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices for this model. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `TrainCore(Matrix<>,Vector<>)` | Performs the model-specific training algorithm. |
| `ValidateOptions(TimeSeriesRegressionOptions<>)` | Validates the provided time series options to ensure they are within acceptable ranges. |
| `ValidateTrainingInputs(Matrix<>,Vector<>)` | Validates the training input data before proceeding with training. |
| `WithParameters(Vector<>)` | Creates a new model with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_autoGuardThreshold` | Auto-scaled guard threshold computed from training data. |
| `_defaultLossFunction` | The default loss function used for gradient computation. |

