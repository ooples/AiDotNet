---
title: "AdversarialRobustnessConfiguration<T, TInput, TOutput>"
description: "Configuration for adversarial robustness and AI safety during model building and inference."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for adversarial robustness and AI safety during model building and inference.

## For Beginners

This is your complete safety and robustness configuration.
You can enable/disable features, customize options, or provide your own implementations.
All settings have sensible defaults based on industry best practices.

## How It Works

This configuration controls all aspects of adversarial robustness and AI safety, replacing
the previous SafetyFilterConfiguration with a unified approach that includes:

- Safety filtering (input/output validation)
- Adversarial attacks and defenses
- Certified robustness
- Content moderation
- Red teaming

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoGenerateModelCardSections` | Gets or sets whether to auto-generate robustness sections in the model card. |
| `CustomAttacks` | Gets or sets custom adversarial attack implementations. |
| `CustomCertifiedDefense` | Gets or sets a custom certified defense implementation. |
| `CustomContentClassifier` | Gets or sets a custom content classifier implementation. |
| `CustomDefense` | Gets or sets a custom adversarial defense implementation. |
| `CustomSafetyFilter` | Gets or sets a custom safety filter implementation. |
| `Enabled` | Gets or sets whether adversarial robustness features are enabled. |
| `EvaluationEpsilons` | Gets or sets the epsilon values to test during robustness evaluation. |
| `IncludeRobustnessInEvaluation` | Gets or sets whether to include robustness evaluation in model evaluation. |
| `IncludeRobustnessMetrics` | Gets or sets whether to include robustness metrics in prediction results. |
| `MinimumCertifiedRadius` | Gets or sets the minimum certified radius required for a prediction to be considered robust. |
| `ModelCardRobustnessNotes` | Gets or sets custom model card robustness notes. |
| `Options` | Gets or sets the robustness options. |
| `RejectNonRobustPredictions` | Gets or sets whether to reject predictions that don't meet the minimum certified radius. |
| `RobustnessEvaluationSampleRatio` | Gets or sets the percentage of test data to use for robustness evaluation. |
| `UseCertifiedInference` | Gets or sets whether to apply certified inference during prediction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BasicSafety` | Creates a configuration with basic safety filtering only. |
| `Comprehensive` | Creates a configuration with comprehensive robustness features. |
| `Disabled` | Creates a disabled configuration (no robustness features). |
| `ForLLM` | Creates a configuration optimized for LLM safety. |
| `WithAdversarialTraining` | Creates a configuration focused on adversarial training. |
| `WithCertification(String)` | Creates a configuration with certified robustness guarantees. |

