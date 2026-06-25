---
title: "EMDDecomposition<T>"
description: "Implements the Empirical Mode Decomposition (EMD) method for time series decomposition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements the Empirical Mode Decomposition (EMD) method for time series decomposition.

## How It Works

**For Beginners:** EMD breaks down a complex signal (like stock prices or temperature readings over time) 
into simpler components called Intrinsic Mode Functions (IMFs). Think of it like separating 
different instruments in a song - you can hear the whole song, but EMD helps you identify 
the individual instruments playing together.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EMDDecomposition(Vector<>,IInterpolation<>,Int32,Double,EMDAlgorithmType)` | Initializes a new instance of the EMD decomposition algorithm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddWhiteNoise(Vector<>,Double)` | Adds random noise to a signal for ensemble methods. |
| `AverageImfs(List<List<Vector<>>>)` | Averages multiple sets of Intrinsic Mode Functions (IMFs) into a single set. |
| `AverageVectors(List<Vector<>>)` | Calculates the average of multiple vectors. |
| `CalculateEnergy(Vector<>)` | Calculates the average energy of a signal. |
| `CalculateResidual(Vector<>,List<Vector<>>)` | Calculates the residual signal by subtracting all IMFs from the original signal. |
| `ComputeEnvelope(Vector<>,EnvelopeType)` | Computes an envelope (upper or lower) for a signal by interpolating between its extrema. |
| `ComputeMeanEnvelope(List<Vector<>>)` | Computes the mean of multiple envelope vectors. |
| `ComputeMultivariateEnvelopes(List<Vector<>>)` | Computes upper and lower envelopes for each projection of a multivariate signal. |
| `Decompose` | Performs the time series decomposition using the selected EMD algorithm. |
| `DecomposeCompleteEnsemble` | Implements the Complete Ensemble EMD (CEEMD) algorithm for enhanced decomposition. |
| `DecomposeEnsemble` | Implements the Ensemble EMD (EEMD) algorithm for more robust decomposition. |
| `DecomposeMultivariate` | Implements Multivariate EMD (MEMD) for decomposing multiple related time series simultaneously. |
| `DecomposeSignal(Vector<>)` | Decomposes a time series signal into its Intrinsic Mode Functions (IMFs). |
| `DecomposeStandard` | Implements the standard EMD algorithm to decompose the time series. |
| `ExtractCEEMDImf(Vector<>,Int32,Double)` | Extracts an Intrinsic Mode Function using the Complete Ensemble Empirical Mode Decomposition method. |
| `ExtractIMF(Vector<>)` | Extracts a single Intrinsic Mode Function from a signal using the sifting process. |
| `ExtractMultivariateIMF(Matrix<>,Int32)` | Extracts Intrinsic Mode Functions from a multivariate (multi-channel) signal. |
| `FindExtrema(Vector<>,EnvelopeType)` | Finds the indices of local maxima or minima in a signal. |
| `IsIMFNegligible(Vector<>)` | Determines if an Intrinsic Mode Function has negligible energy. |
| `IsLocalMaximum(Vector<>,Int32)` | Determines if a point in a signal is a local maximum. |
| `IsLocalMinimum(Vector<>,Int32)` | Determines if a point in a signal is a local minimum. |
| `IsMeanEnvelopeNearZero(Vector<>,Vector<>)` | Determines if the difference between two IMFs is small enough to consider the sifting process complete. |
| `IsMultivariateMeanEnvelopeNearZero(Matrix<>,Matrix<>)` | Determines if the difference between two multivariate IMFs is small enough to consider the sifting process complete. |
| `IsMultivariateResidual(Matrix<>)` | Checks if a multivariate signal meets the criteria to be considered a residual. |
| `IsResidual(Vector<>)` | Determines if a signal meets the criteria to be considered a residual. |
| `ProjectSignal(Matrix<>,Int32)` | Projects a multivariate signal onto different directions to analyze its components. |

