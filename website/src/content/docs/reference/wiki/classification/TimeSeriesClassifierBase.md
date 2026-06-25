---
title: "TimeSeriesClassifierBase<T>"
description: "Base class for time series classification models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.TimeSeries`

Base class for time series classification models.

## For Beginners

This base class provides common functionality for time series
classifiers. It handles sequence-based input (3D tensors) and provides the infrastructure
for training on time series data while inheriting all the classification machinery from
ClassifierBase.

## How It Works

**Key concepts:**

- **Sequence:** A time-ordered series of observations (e.g., 100 time steps)
- **Channel:** A variable measured at each time step (e.g., x, y, z accelerometer)
- **Sample:** One complete time series with its class label

**Input format:** Sequences are passed as 3D tensors with shape:
[num_samples, sequence_length, num_channels]

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesClassifierBase(TimeSeriesClassifierOptions<>)` | Creates a new time series classifier base. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumChannels` | Gets or sets the number of channels (variables) in the time series. |
| `SequenceLength` | Gets or sets the expected sequence length. |
| `SupportsVariableLengths` | Gets whether this classifier supports variable-length sequences. |
| `TimeSeriesOptions` | Gets the time series classifier options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FlattenSequences(Tensor<>)` | Flattens 3D sequence data into 2D matrix format for standard classifier processing. |
| `PredictSequenceProbabilities(Tensor<>)` | Predicts class probabilities for a batch of time series sequences. |
| `PredictSequences(Tensor<>)` | Predicts class labels for a batch of time series sequences. |
| `TrainOnSequences(Tensor<>,Vector<>)` | Trains the classifier on a collection of time series sequences. |
| `UnflattenToSequences(Matrix<>)` | Converts 2D flattened data back to 3D sequence format. |
| `ValidateSequenceInput(Tensor<>,Vector<>)` | Validates the input sequences have the correct shape and dimensions. |

