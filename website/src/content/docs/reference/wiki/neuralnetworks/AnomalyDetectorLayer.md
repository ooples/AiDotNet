---
title: "AnomalyDetectorLayer<T>"
description: "Represents a layer that detects anomalies by comparing predictions with actual inputs."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that detects anomalies by comparing predictions with actual inputs.

## For Beginners

This layer identifies patterns that don't match what the network expected.

Think of anomaly detection like this:

- The network learns what "normal" looks like from the data
- This layer compares new inputs to what the network expected to see
- If the actual input is very different from the prediction, it's flagged as anomalous
- The output is an "anomaly score" between 0 and 1 (higher means more unusual)

For example, in monitoring network traffic, the system might learn normal patterns
and then use this layer to alert when unusual activity is detected that might
indicate a security breach.

## How It Works

The AnomalyDetectionLayer compares the predicted state with the actual state to calculate an anomaly score.
This score represents how unexpected or surprising the current input is, given the model's predictions.
Higher scores indicate more anomalous (unexpected) inputs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnomalyDetectorLayer(Int32,Double,Int32,Double,IEngine)` | Initializes a new instance of the `AnomalyDetectorLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | The computation engine (CPU or GPU) for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAnomalyScores(Tensor<>,Tensor<>)` | Calculates the anomaly score based on the difference between actual and predicted states. |
| `Forward(Tensor<>)` | Performs the forward pass of the anomaly detection layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU acceleration. |
| `GetAnomalyScore` | Gets the current anomaly score. |
| `GetAnomalyStatistics` | Gets the statistical properties of recent anomaly scores. |
| `GetParameters` | Gets all parameters of the layer as a single vector. |
| `IsAnomaly` | Determines if the current input is anomalous based on the anomaly score. |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateAnomalyHistory(Double)` | Updates the history of anomaly scores. |
| `UpdateParameters()` | Updates the parameters of the layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_anomalyHistory` | The history of recent anomaly scores for adaptive thresholding. |
| `_anomalyThreshold` | The threshold for determining anomalous inputs based on the anomaly score. |
| `_historyCapacity` | The maximum number of anomaly scores to keep in history. |
| `_lastInputShape` | Stores the most recent input shape for any-rank tensor support. |
| `_smoothedAnomalyScore` | The current smoothed anomaly score. |
| `_smoothingFactor` | The smoothing factor for exponential moving average of anomaly scores. |

