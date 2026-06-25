---
title: "HodrickPrescottDecomposition<T>"
description: "Implements the Hodrick-Prescott filter for decomposing time series data into trend and cyclical components."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements the Hodrick-Prescott filter for decomposing time series data into trend and cyclical components.

## How It Works

**For Beginners:** The Hodrick-Prescott filter is a mathematical tool that helps separate a time series 
(like stock prices or economic data over time) into two parts:

1. A smooth trend component that shows the long-term direction
2. A cyclical component that shows short-term fluctuations around the trend

Think of it like separating a bumpy road (your data) into the general path (trend) 
and the bumps along the way (cycles).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HodrickPrescottDecomposition(Vector<>,Double,IMatrixDecomposition<>,HodrickPrescottAlgorithmType)` | Initializes a new instance of the Hodrick-Prescott decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConstructSecondDifferenceMatrix(Int32)` | Constructs a second difference matrix used in the matrix-based Hodrick-Prescott filter. |
| `Decompose` | Performs the time series decomposition using the selected algorithm. |
| `DecomposeFrequencyDomainMethod` | Decomposes the time series using frequency domain analysis with Fast Fourier Transform. |
| `DecomposeIterativeMethod` | Decomposes the time series using an iterative approach to the Hodrick-Prescott filter. |
| `DecomposeKalmanFilterMethod` | Decomposes the time series using a Kalman filter approach. |
| `DecomposeMatrixMethod` | Decomposes the time series using the standard matrix-based Hodrick-Prescott filter. |
| `DecomposeStateSpaceMethod` | Decomposes the time series using a state space modeling approach. |
| `DecomposeWaveletMethod` | Decomposes the time series using wavelet transform. |
| `DiscreteWaveletTransform(Vector<>,Int32)` | Performs a discrete wavelet transform on the input data. |
| `InverseDiscreteWaveletTransform(Vector<>,Int32)` | Performs an inverse discrete wavelet transform to reconstruct the original signal. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_algorithm` | The algorithm type to use for the Hodrick-Prescott decomposition. |
| `_decomposition` | Optional matrix decomposition method used for solving the linear system in the matrix method. |
| `_lambda` | The smoothing parameter that controls the balance between smoothness of the trend and fit to the data. |

