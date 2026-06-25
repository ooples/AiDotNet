---
title: "RocketClassifier<T>"
description: "Implements ROCKET (Random Convolutional Kernel Transform) for time series classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.TimeSeries`

Implements ROCKET (Random Convolutional Kernel Transform) for time series classification.

## For Beginners

ROCKET is a highly efficient and accurate time series classifier
that uses thousands of random convolutional kernels to extract features. Despite using random
kernels (no training), it achieves state-of-the-art accuracy while being orders of magnitude
faster than other methods.

## How It Works

**How ROCKET works:**

- Generate thousands of random convolutional kernels with varying lengths, dilations, and weights
- Apply each kernel to the input time series to produce an output array
- Extract two features from each kernel output: max value and proportion of positive values (PPV)
- Use these features with a simple linear classifier (e.g., Ridge regression)

**Why ROCKET is so effective:**

- Random kernels + large quantity = covers diverse patterns
- PPV captures frequency of pattern occurrence
- Max value captures pattern strength
- Dilation handles different time scales

**Reference:** Dempster et al., "ROCKET: Exceptionally fast and accurate time series classification
using random convolutional kernels" (2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RocketClassifier(RocketOptions<>)` | Creates a new ROCKET classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumChannels` | Gets or sets the number of channels (variables) in the time series. |
| `SequenceLength` | Gets or sets the expected sequence length. |
| `SupportsVariableLengths` | Gets whether this classifier supports variable-length sequences. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ApplyKernel(Double[],RocketClassifier<>.RocketKernel)` | Applies a single kernel and extracts features. |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `ComputeRidgeWeights(Matrix<>,Vector<>,Double)` | Computes ridge regression weights. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `GenerateKernels(Int32)` | Generates the random convolutional kernels. |
| `GetParameters` |  |
| `Predict(Matrix<>)` | Predicts class labels for feature matrix. |
| `PredictSequenceProbabilities(Tensor<>)` | Predicts class probabilities for time series sequences. |
| `PredictSequences(Tensor<>)` | Predicts class labels for time series sequences. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `SolveLinearSystem(Matrix<>,Vector<>)` | Solves a linear system Ax = b using Gaussian elimination. |
| `Train(Matrix<>,Vector<>)` | Trains the classifier on feature matrix (used after transform). |
| `TrainBinaryClassifier(Matrix<>,Vector<>)` | Trains a binary classifier using ridge regression approach. |
| `TrainMultiClassClassifier(Matrix<>,Vector<>)` | Trains a multi-class classifier using one-vs-rest approach. |
| `TrainOnSequences(Tensor<>,Vector<>)` | Trains the ROCKET classifier on time series data. |
| `TransformSequences(Tensor<>)` | Transforms sequences into feature vectors using ROCKET kernels. |
| `ValidateSequenceInput(Tensor<>,Vector<>)` | Validates the input sequences. |
| `WithParameters(Vector<>)` |  |

