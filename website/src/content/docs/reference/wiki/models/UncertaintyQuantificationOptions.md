---
title: "UncertaintyQuantificationOptions"
description: "Configuration options for enabling uncertainty quantification during inference."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for enabling uncertainty quantification during inference.

## For Beginners

This lets you ask the model not only "what is the prediction?"
but also "how sure are you?"

## How It Works

Uncertainty quantification (UQ) augments standard point predictions with an uncertainty estimate.
For supported model types, the library can sample multiple stochastic predictions and aggregate them
into a mean prediction and an uncertainty estimate (variance).

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptiveConformalBins` | Gets or sets the number of confidence bins used when `ConformalMode` is `Adaptive`. |
| `CalibrationMethod` | Gets or sets the probability calibration method used when calibration labels are provided. |
| `ConformalConfidenceLevel` | Gets or sets the desired conformal coverage level when using conformal prediction. |
| `ConformalMode` | Gets or sets the conformal calibration mode used when producing conformal intervals/sets. |
| `CrossConformalFolds` | Gets or sets the number of folds used when `ConformalMode` is `CrossConformal`. |
| `DeepEnsembleInitialNoiseStdDev` | Gets or sets the standard deviation of the initial parameter perturbation applied when constructing ensemble members. |
| `DeepEnsembleSize` | Gets or sets the number of independently trained models used for deep ensemble uncertainty estimation. |
| `DenormalizeUncertainty` | Gets or sets whether the returned uncertainty should be denormalized to match the output scale. |
| `EnableIsotonicRegressionCalibration` | Gets or sets whether to fit and apply isotonic regression calibration (binary calibration) when calibration labels are provided. |
| `EnablePlattScaling` | Gets or sets whether to fit and apply Platt scaling (binary calibration) when calibration labels are provided. |
| `EnableTemperatureScaling` | Gets or sets whether to fit and apply temperature scaling for classification-like outputs when calibration labels are provided. |
| `Enabled` | Gets or sets whether uncertainty quantification is enabled. |
| `LaplacePriorPrecision` | Gets or sets the prior precision (inverse variance) used by diagonal Laplace approximation. |
| `Method` | Gets or sets the uncertainty quantification strategy to use. |
| `MonteCarloDropoutRate` | Gets or sets the dropout rate used when injecting Monte Carlo Dropout layers automatically. |
| `NumSamples` | Gets or sets the number of stochastic samples to draw when using sampling-based methods. |
| `PosteriorFitMaxSamples` | Gets or sets the maximum number of samples used to fit Laplace/SWAG posteriors from calibration data. |
| `RandomSeed` | Gets or sets an optional random seed for reproducible Monte Carlo sampling. |
| `SwagBurnInSteps` | Gets or sets the number of initial SWAG steps to skip before collecting snapshots. |
| `SwagLearningRate` | Gets or sets the learning rate used for SWAG posterior fitting on calibration data. |
| `SwagNumSnapshots` | Gets or sets the number of SWAG snapshots to collect when fitting a SWAG posterior. |
| `SwagNumSteps` | Gets or sets the number of SWAG update steps used to collect snapshots. |

