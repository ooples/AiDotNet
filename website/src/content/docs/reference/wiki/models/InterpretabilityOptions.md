---
title: "InterpretabilityOptions"
description: "Configuration options for model interpretability and explainability features."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for model interpretability and explainability features.

## For Beginners

These options control how your model explains its predictions.

Why interpretability matters:

- Understanding why a model makes certain predictions builds trust
- Identifying which features drive predictions helps validate model logic
- Explaining individual predictions is often required for regulatory compliance
- Finding biases in feature importance can reveal unfair model behavior

Available explanation methods:

**Model-Agnostic Methods** (work with any model):

- **SHAP**: Shows how each feature contributed to each prediction (local and global)
- **LIME**: Explains individual predictions using simple approximations
- **Permutation Importance**: Shows which features matter most overall
- **Partial Dependence**: Shows how features affect predictions on average
- **Feature Interaction**: Measures how features interact using H-statistic
- **Counterfactual**: Shows what would need to change for a different prediction
- **Anchor**: Creates rule-based explanations
- **Global Surrogate**: Trains a simple model to mimic the complex one

**Neural Network Methods** (require gradient access):

- **Integrated Gradients**: Theoretically-grounded attribution method
- **DeepLIFT**: Fast attribution comparing to baseline
- **GradCAM**: Visual heatmaps for CNN explanations

## How It Works

This class configures which interpretability methods are available and how they behave.
It supports multiple model-agnostic explanation techniques that work with any model,
as well as specialized neural network explanation methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `DeepLIFTUseRevealCancel` | Gets or sets whether to use the RevealCancel rule for DeepLIFT (default: Rescale). |
| `EnableAnchor` | Gets or sets whether Anchor explanations are enabled. |
| `EnableCounterfactual` | Gets or sets whether Counterfactual explanations are enabled. |
| `EnableDeepLIFT` | Gets or sets whether DeepLIFT is enabled for neural network explanation. |
| `EnableFeatureInteraction` | Gets or sets whether Feature Interaction analysis (H-statistic) is enabled. |
| `EnableGlobalSurrogate` | Gets or sets whether Global Surrogate modeling is enabled. |
| `EnableGradCAM` | Gets or sets whether GradCAM is enabled for CNN visual explanations. |
| `EnableIntegratedGradients` | Gets or sets whether Integrated Gradients is enabled for neural network explanation. |
| `EnableLIME` | Gets or sets whether LIME explanations are enabled. |
| `EnablePartialDependence` | Gets or sets whether Partial Dependence Plots (PDP) are enabled. |
| `EnablePermutationImportance` | Gets or sets whether Permutation Feature Importance is enabled. |
| `EnableSHAP` | Gets or sets whether SHAP explanations are enabled. |
| `EnableTreeSHAP` | Gets or sets whether TreeSHAP is enabled for tree-based model explanation. |
| `FeatureNames` | Gets or sets optional feature names for more readable explanations. |
| `IntegratedGradientsSteps` | Gets or sets the number of integration steps for Integrated Gradients. |
| `LIMEKernelWidth` | Gets or sets the kernel width for LIME (controls locality). |
| `LIMESampleCount` | Gets or sets the number of perturbed samples for LIME. |
| `MaxBackgroundSamples` | Gets or sets the maximum number of background samples for SHAP baseline calculation. |
| `PermutationRepeatCount` | Gets or sets the number of times to repeat permutation for more stable estimates. |
| `RandomSeed` | Gets or sets the random seed for reproducible explanations. |
| `SHAPSampleCount` | Gets or sets the number of samples to use for Kernel SHAP approximation. |
| `UseGradCAMPlusPlus` | Gets or sets whether to use GradCAM++ instead of standard GradCAM. |

