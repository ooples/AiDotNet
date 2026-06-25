---
title: "UncertaintyOptions"
description: "Configuration options for uncertainty quantification."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for uncertainty quantification.

## For Beginners

Uncertainty tells you how confident the model is:

- **Aleatoric uncertainty:** Inherent noise in the data (can't be reduced)
- **Epistemic uncertainty:** Model's lack of knowledge (can be reduced with more data)

A good uncertainty estimate lets you flag predictions the model isn't sure about,
which might need human review.

## How It Works

Uncertainty quantification measures how confident a model is in its predictions.
This is crucial for high-stakes applications where knowing "I don't know" is valuable.

## Properties

| Property | Summary |
|:-----|:--------|
| `CWCPenaltyFactor` | Penalty factor for CWC calculation. |
| `CalibrateUncertainty` | Whether to compute calibrated uncertainty. |
| `CalibrationMethod` | Uncertainty calibration method. |
| `CheckQuantileCrossing` | Whether to check for quantile crossing. |
| `ComputeCWC` | Whether to compute CWC (Coverage Width Criterion). |
| `ComputeMPIW` | Whether to compute MPIW (Mean Prediction Interval Width). |
| `ComputeMutualInformation` | Whether to compute mutual information (epistemic uncertainty). |
| `ComputePICP` | Whether to compute PICP (Prediction Interval Coverage Probability). |
| `ComputePredictionIntervals` | Whether to compute prediction intervals. |
| `ComputePredictiveEntropy` | Whether to compute predictive entropy. |
| `ComputeSharpness` | Whether to compute sharpness score. |
| `ComputeUncertaintyReliability` | Whether to compute reliability diagrams for uncertainty. |
| `DecomposeUncertainty` | Whether to decompose uncertainty into aleatoric/epistemic. |
| `DecompositionMethod` | Method for uncertainty decomposition. |
| `EnsembleSize` | Number of ensemble members for ensemble-based uncertainty. |
| `HighUncertaintyThreshold` | Threshold for high uncertainty (percentile). |
| `IdentifyHighUncertaintySamples` | Whether to identify high-uncertainty samples. |
| `IntervalMethod` | Method for prediction interval estimation. |
| `MCDropoutRate` | Dropout rate for MC Dropout. |
| `MCDropoutSamples` | Number of Monte Carlo dropout samples. |
| `PredictionIntervalCoverage` | Coverage level for prediction intervals. |
| `Quantiles` | Quantiles to compute for quantile regression. |
| `RandomSeed` | Random seed for reproducibility. |
| `ReliabilityBins` | Number of bins for uncertainty reliability diagram. |

