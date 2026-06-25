---
title: "AdversarialDefenseOptions<T>"
description: "Configuration options for adversarial defense mechanisms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for adversarial defense mechanisms.

## For Beginners

These settings control how your "armor" protects the AI model.
You can adjust how the defense is applied, how strong it should be, and what techniques to use.

## How It Works

These options control how models are defended against adversarial attacks through
training procedures, preprocessing, and ensemble methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdversarialRatio` | Gets or sets the ratio of adversarial examples to include in training. |
| `AttackMethod` | Gets or sets the attack method to use during adversarial training. |
| `EnsembleSize` | Gets or sets the number of models in the ensemble. |
| `Epsilon` | Gets or sets the perturbation budget for adversarial training. |
| `PreprocessingMethod` | Gets or sets the preprocessing method to use. |
| `TrainingEpochs` | Gets or sets the number of training epochs. |
| `UseEnsemble` | Gets or sets whether to use ensemble defenses. |
| `UsePreprocessing` | Gets or sets whether to use input preprocessing for defense. |

