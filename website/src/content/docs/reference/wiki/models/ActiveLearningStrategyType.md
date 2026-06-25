---
title: "ActiveLearningStrategyType"
description: "Specifies the active learning strategy to use for sample selection."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the active learning strategy to use for sample selection.

## For Beginners

Active learning strategies help identify which unlabeled samples
would be most valuable to label. Different strategies use different criteria to measure
how "informative" a sample is for training the model.

## Fields

| Field | Summary |
|:-----|:--------|
| `BALD` | Bayesian Active Learning by Disagreement - uses mutual information between predictions and model parameters for selection. |
| `BatchBALD` | Batch-mode BALD that accounts for redundancy when selecting multiple samples. |
| `CoreSetSelection` | Selects representative samples that form a core set of the data. |
| `DensityWeightedSampling` | Weights uncertainty by sample density in the input space. |
| `DiversitySampling` | Selects samples that maximize coverage of the input space. |
| `EntropySampling` | Selects samples based on entropy of predicted class probabilities. |
| `ExpectedModelChange` | Selects samples that would cause the largest change to model parameters. |
| `HybridSampling` | Combines multiple criteria (uncertainty + diversity) for selection. |
| `InformationDensity` | Combines uncertainty with representativeness based on local density. |
| `LeastConfidenceSampling` | Selects samples where the top prediction has the lowest confidence. |
| `MarginSampling` | Selects samples with the smallest margin between top two predictions. |
| `QueryByCommittee` | Uses multiple models and selects samples where they disagree the most. |
| `Random` | Selects samples randomly. |
| `UncertaintySampling` | Selects samples where the model is most uncertain about predictions. |
| `VariationRatios` | Uses variation ratios (1 - max probability) for uncertainty estimation. |

