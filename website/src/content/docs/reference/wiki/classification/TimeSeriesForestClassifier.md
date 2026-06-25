---
title: "TimeSeriesForestClassifier<T>"
description: "Implements the Time Series Forest classifier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.TimeSeries`

Implements the Time Series Forest classifier.

## For Beginners

Time Series Forest builds an ensemble of decision trees, where each
tree is trained on features extracted from a randomly selected interval of the time series.
This approach captures patterns at different time scales and positions.

## How It Works

**How it works:**

- For each tree, randomly select an interval (start, end) from the time series
- Extract summary features from that interval (mean, std, slope)
- Train a decision tree on these interval features
- Repeat for all trees in the ensemble
- Predict by majority voting across all trees

**Key features:**

- Captures local patterns at different positions in the sequence
- Robust to noise through ensemble averaging
- Interpretable through interval selection
- Handles variable-length sequences naturally

**Reference:** Deng et al., "A Time Series Forest for Classification and Feature Extraction" (2013)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesForestClassifier(TimeSeriesForestOptions<>)` | Creates a new Time Series Forest classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumChannels` | Gets the number of channels (variables) in the time series. |
| `SequenceLength` | Gets the expected sequence length. |
| `SupportsVariableLengths` | Gets whether this classifier supports variable-length sequences. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `Predict(Matrix<>)` |  |
| `PredictSequenceProbabilities(Tensor<>)` | Predicts class probabilities for time series sequences. |
| `PredictSequences(Tensor<>)` | Predicts class labels for time series sequences. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `TrainOnSequences(Tensor<>,Vector<>)` | Trains the Time Series Forest on time series sequences. |
| `WithParameters(Vector<>)` |  |

