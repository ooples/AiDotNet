---
title: "MiniRocketClassifier<T>"
description: "Implements MiniRocket for time series classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.TimeSeries`

Implements MiniRocket for time series classification.

## For Beginners

MiniRocket is a faster, simpler version of ROCKET that uses
deterministic kernels with fixed weights. It achieves similar accuracy to ROCKET while
being much faster and more memory efficient.

## How It Works

**Key differences from ROCKET:**

- Uses only PPV (proportion of positive values) features, not max values
- Kernels have fixed weights from {-1, 2} (instead of random weights)
- Uses a fixed set of 84 base kernels (instead of random kernels)
- Biases are computed from quantiles of convolution outputs

**How MiniRocket works:**

- Define 84 deterministic kernel patterns using weights from {-1, 2}
- For each kernel, compute convolution at multiple dilations
- Compute bias values from quantiles of convolution outputs
- Extract PPV features using each (kernel, dilation, bias) combination
- Train a linear classifier on the extracted features

**Reference:** Dempster et al., "MiniRocket: A Very Fast (Almost) Deterministic Transform
for Time Series Classification" (2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MiniRocketClassifier(MiniRocketOptions<>)` | Creates a new MiniRocket classifier. |

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
| `TrainOnSequences(Tensor<>,Vector<>)` | Trains the MiniRocket classifier on time series sequences. |
| `WithParameters(Vector<>)` |  |

