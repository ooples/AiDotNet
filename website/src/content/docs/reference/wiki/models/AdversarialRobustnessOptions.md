---
title: "AdversarialRobustnessOptions<T>"
description: "Unified configuration options for adversarial robustness and AI safety."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Unified configuration options for adversarial robustness and AI safety.

## For Beginners

This is your one-stop shop for all AI safety and robustness settings.
You can configure how strictly inputs are validated, how the model is protected against attacks,
and what guarantees you want about model predictions.

## How It Works

This comprehensive options class consolidates all adversarial robustness settings including:

- Safety filtering (input validation, output filtering, harmful content detection)
- Adversarial attacks (for robustness testing and evaluation)
- Adversarial defenses (training and preprocessing)
- Certified robustness (provable guarantees)
- Content moderation (for LLM applications)

## Properties

| Property | Summary |
|:-----|:--------|
| `AdversarialTrainingRatio` | Gets or sets the ratio of adversarial examples to include in training. |
| `AttackEpsilon` | Gets or sets the maximum perturbation budget (epsilon) for attacks. |
| `AttackIterations` | Gets or sets the number of iterations for iterative attacks. |
| `AttackMethods` | Gets or sets which attack methods to use for testing. |
| `AttackNormType` | Gets or sets the norm type for perturbation constraints. |
| `AttackStepSize` | Gets or sets the step size for iterative attacks. |
| `BatchSize` | Gets or sets the batch size for robustness operations. |
| `BlockPromptInjections` | Gets or sets whether to detect and block prompt injection attacks. |
| `CertificationConfidence` | Gets or sets the confidence level for certification. |
| `CertificationMethod` | Gets or sets the certification method to use. |
| `CertificationNoiseSigma` | Gets or sets the noise standard deviation for randomized smoothing. |
| `CertificationNormType` | Gets or sets the norm type for certification. |
| `CertificationSamples` | Gets or sets the number of samples for randomized smoothing. |
| `DefenseEpsilon` | Gets or sets the perturbation budget for adversarial training. |
| `EnableAdversarialTraining` | Gets or sets whether to enable adversarial training. |
| `EnableCertifiedRobustness` | Gets or sets whether to enable certified robustness. |
| `EnableContentModeration` | Gets or sets whether to enable content moderation for LLM outputs. |
| `EnableFactualityChecking` | Gets or sets whether to enable factuality checking. |
| `EnableInputValidation` | Gets or sets whether to enable input validation. |
| `EnableOutputFiltering` | Gets or sets whether to enable output filtering. |
| `EnableRedTeaming` | Gets or sets whether to enable red teaming during evaluation. |
| `EnableRobustnessTesting` | Gets or sets whether to enable adversarial robustness testing. |
| `EnableSafetyFiltering` | Gets or sets whether safety filtering is enabled. |
| `EnsembleSize` | Gets or sets the number of models in the ensemble. |
| `FilterPII` | Gets or sets whether to filter PII (personally identifiable information). |
| `HallucinationThreshold` | Gets or sets the hallucination detection threshold. |
| `HarmfulContentCategories` | Gets or sets the harmful content categories to check for. |
| `JailbreakSensitivity` | Gets or sets the jailbreak detection sensitivity. |
| `LogFilteredContent` | Gets or sets whether to log filtered content for review. |
| `MaxInputLength` | Gets or sets the maximum input length to process. |
| `PIITypes` | Gets or sets the types of PII to filter. |
| `PreprocessingMethod` | Gets or sets the preprocessing method to use. |
| `PromptInjectionSensitivity` | Gets or sets the prompt injection detection sensitivity. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `RedTeamingCategories` | Gets or sets the red teaming categories to test. |
| `RedTeamingPromptCount` | Gets or sets the number of red teaming prompts to generate. |
| `SafetyLogFilePath` | Gets or sets the file path for filtered content logging. |
| `SafetyThreshold` | Gets or sets the safety threshold for content filtering. |
| `TargetClass` | Gets or sets the target class for targeted attacks. |
| `TrainingAttackMethod` | Gets or sets the attack method to use during adversarial training. |
| `UseContentClassifier` | Gets or sets whether to use a classifier for harmful content detection. |
| `UseEnsembleDefense` | Gets or sets whether to use ensemble defenses. |
| `UseInputPreprocessing` | Gets or sets whether to use input preprocessing for defense. |
| `UseRandomStartForAttacks` | Gets or sets whether to use random initialization for attacks. |
| `UseTargetedAttacks` | Gets or sets whether to use targeted attacks. |
| `UseTightCertificationBounds` | Gets or sets whether to use tight bounds computation. |
| `VerboseLogging` | Gets or sets whether to enable verbose logging. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdversarialTrainingFocus` | Creates options for adversarial training focus. |
| `BasicSafety` | Creates options for basic safety filtering only. |
| `ComprehensiveRobustness` | Creates options for comprehensive robustness with certified guarantees. |
| `LLMSafety` | Creates options for LLM safety with content moderation. |

