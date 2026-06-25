---
title: "RobustnessOptions"
description: "Configuration options for robustness evaluation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Evaluation.Options`

Configuration options for robustness evaluation.

## For Beginners

Robustness tests answer questions like:

- What happens if there's noise in the input? (Noise robustness)
- What if someone intentionally tries to fool the model? (Adversarial robustness)
- What if the input has small errors or missing values? (Perturbation robustness)

This is important for real-world deployments where inputs aren't always clean.

## How It Works

Robustness evaluation measures how well a model performs under input perturbations,
noise, and adversarial attacks. A robust model maintains performance despite variations.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdversarialEpsilon` | Epsilon for adversarial attacks. |
| `AttackIterations` | Number of attack iterations (for iterative methods). |
| `AttackMethod` | Adversarial attack method. |
| `ComputeFlipRate` | Whether to compute prediction flip rate under perturbation. |
| `ComputeInputGradients` | Whether to compute input gradient norms. |
| `DropoutRates` | Feature dropout rates to test (random feature removal). |
| `EstimateLipschitzConstant` | Whether to estimate local Lipschitz constant. |
| `FeaturesToPerturb` | Specific features to perturb. |
| `LipschitzSamples` | Number of samples for Lipschitz estimation. |
| `MaxDegreeOfParallelism` | Maximum parallelism. |
| `MaxSamples` | Maximum samples to test. |
| `MissingStrategy` | Missing value handling strategy. |
| `MissingValueRates` | Missing value rates to test. |
| `NoiseLevels` | Noise levels (standard deviations) to test. |
| `NoiseSamplesPerInput` | Number of noise samples per input. |
| `OutlierRates` | Outlier contamination rates to test. |
| `ParallelExecution` | Whether to run tests in parallel. |
| `PerturbationMagnitude` | Perturbation magnitude (relative). |
| `RandomSeed` | Random seed for reproducibility. |
| `SampleRatio` | Subset of samples to test (ratio). |
| `TestAdversarialRobustness` | Whether to test adversarial robustness. |
| `TestFeaturePerturbation` | Whether to test feature perturbation impact. |
| `TestGaussianNoise` | Whether to test Gaussian noise robustness. |
| `TestMissingValues` | Whether to test missing value robustness. |
| `TestOutlierRobustness` | Whether to test outlier robustness. |
| `TestUniformNoise` | Whether to test uniform noise. |

