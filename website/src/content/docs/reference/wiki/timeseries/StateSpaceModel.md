---
title: "StateSpaceModel<T>"
description: "Implements a State Space Model for time series analysis and forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements a State Space Model for time series analysis and forecasting.

## For Beginners

A State Space Model is like tracking the position of a moving object when you can only see its shadow.
The actual position (state) is hidden, but you can observe its effects (the shadow).

For example, if you're tracking the economy, you might not directly observe the "true state" of the economy,
but you can see indicators like GDP, unemployment rates, etc. The State Space Model helps infer the hidden
state from these observations and predict future values.

The model has two main components:

1. A transition equation that describes how the hidden state evolves over time
2. An observation equation that relates the hidden state to what we actually observe

This implementation uses the Kalman filter and smoother algorithms to estimate the hidden states
and learn the model parameters from data.

## How It Works

State Space Models represent time series data as a system with hidden states that evolve over time
according to probabilistic rules. They are powerful tools for modeling complex dynamic systems
and can handle missing data, multiple variables, and non-stationary patterns.

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
    .ConfigureModel(new StateSpaceModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"StateSpaceModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StateSpaceModel` | Initializes a new instance with default settings. |
| `StateSpaceModel(StateSpaceModelOptions<>)` | Initializes a new instance of the StateSpaceModel class with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateMatrixDifference(Matrix<>,Matrix<>)` | Calculates the Frobenius norm of the difference between two matrices. |
| `CheckConvergence` | Checks if the model parameters have converged during training. |
| `CreateInstance` | Creates a new instance of the State Space Model with the same options. |
| `DeserializeCore(BinaryReader)` | Deserializes the model's core parameters from a binary reader. |
| `EvaluateModel(Matrix<>,Vector<>)` | Evaluates the performance of the trained model on test data. |
| `GetModelMetadata` | Gets metadata about the model, including its type, parameters, and configuration. |
| `KalmanFilter(Matrix<>)` | Applies the Kalman filter to estimate the hidden states based on observations. |
| `KalmanSmoother(List<Vector<>>,List<Vector<>>,List<Matrix<>>,List<Matrix<>>)` | Applies the Kalman smoother to refine the state estimates using all observations. |
| `Predict(Matrix<>)` | Generates predictions using the trained state space model. |
| `PredictSingle(Vector<>)` | Predicts a single value based on the input vector. |
| `SerializeCore(BinaryWriter)` | Serializes the model's core parameters to a binary writer. |
| `TrainCore(Matrix<>,Vector<>)` | Core implementation of the training logic for the State Space Model. |
| `UpdateParameters(Matrix<>,List<Vector<>>,List<Matrix<>>)` | Updates the model parameters based on the estimated states. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_convergenceThreshold` | The threshold for determining when the parameter updates have converged. |
| `_initialState` | The initial state vector at time t=0. |
| `_learningRate` | The learning rate for parameter updates during training. |
| `_maxIterations` | The maximum number of iterations for the EM algorithm during training. |
| `_observationMatrix` | The observation matrix that relates the hidden state to the observed measurements. |
| `_observationNoise` | The covariance matrix of the observation noise, representing measurement uncertainty. |
| `_observationSize` | The dimension of the observation vector. |
| `_previousObservationMatrix` | The observation matrix from the previous iteration, used to check convergence. |
| `_previousTransitionMatrix` | The transition matrix from the previous iteration, used to check convergence. |
| `_processNoise` | The covariance matrix of the process noise, representing uncertainty in the state transition. |
| `_smoothedStates` | The smoothed states from training, used for in-sample predictions. |
| `_stateSize` | The dimension of the state vector. |
| `_tolerance` | The convergence tolerance for the EM algorithm. |
| `_transitionMatrix` | The state transition matrix that describes how the hidden state evolves from one time step to the next. |

