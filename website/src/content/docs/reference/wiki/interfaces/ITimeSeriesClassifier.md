---
title: "ITimeSeriesClassifier<T>"
description: "Interface for time series classification models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for time series classification models.

## For Beginners

Time series classification assigns labels to entire sequences
rather than individual data points. For example, classifying an ECG recording as "normal"
or "abnormal", or classifying a gesture based on accelerometer data.

## How It Works

**How time series classification differs from regular classification:**

- Input is a sequence (time series) not a single feature vector
- Order of observations matters
- May have multiple channels (multivariate time series)
- Sequences can have different lengths

**Common Approaches:**

- **Distance-based:** DTW + 1-NN, Shapelet-based
- **Feature extraction:** ROCKET, MiniRocket, TSFresh
- **Deep learning:** CNN, LSTM, Transformers
- **Ensemble:** Time Series Forest, BOSS ensemble

**Applications:**

- Medical diagnosis (ECG, EEG analysis)
- Human activity recognition (accelerometer data)
- Speech/audio classification
- Anomaly detection in sensor data
- Gesture recognition

## Properties

| Property | Summary |
|:-----|:--------|
| `NumChannels` | Gets the number of channels (variables) in the time series. |
| `SequenceLength` | Gets the expected sequence length for input time series. |
| `SupportsVariableLengths` | Gets whether the classifier can handle variable-length sequences. |

## Methods

| Method | Summary |
|:-----|:--------|
| `PredictSequenceProbabilities(Tensor<>)` | Predicts class probabilities for a batch of time series sequences. |
| `PredictSequences(Tensor<>)` | Predicts class labels for a batch of time series sequences. |
| `TrainOnSequences(Tensor<>,Vector<>)` | Trains the classifier on a collection of time series sequences. |

