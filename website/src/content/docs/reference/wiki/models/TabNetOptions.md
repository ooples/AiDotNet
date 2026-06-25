---
title: "TabNetOptions<T>"
description: "Configuration options for TabNet, an attention-based interpretable deep learning model for tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabNet, an attention-based interpretable deep learning model for tabular data.

## For Beginners

TabNet is a neural network specifically designed for tables of data
(like spreadsheets or databases). Unlike traditional neural networks that use all features
at once, TabNet learns to focus on the most important features for each prediction.

Key advantages of TabNet:

- **Interpretable**: You can see which features the model uses for each prediction
- **Feature Selection**: Automatically selects relevant features at each step
- **No Preprocessing**: Works directly with numerical and categorical data
- **Competitive Performance**: Often matches or beats gradient boosting methods

Example usage:

## How It Works

TabNet uses sequential attention to choose which features to reason from at each decision step,
enabling interpretable feature selection while achieving performance competitive with gradient boosting.

Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, AAAI 2021)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchNormalizationMomentum` | Gets or sets the momentum for batch normalization. |
| `CategoricalEmbeddingDimension` | Gets or sets the embedding dimension for categorical features. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EnableGradientClipping` | Gets or sets whether to clip gradients. |
| `EnablePreTraining` | Gets or sets whether to use pre-training (encoder-decoder) mode. |
| `Epsilon` | Gets or sets the epsilon value for numerical stability in batch normalization. |
| `FeatureDimension` | Gets or sets the dimension of the feature transformation (also known as HiddenDimension). |
| `HiddenDimension` | Alias for FeatureDimension for backward compatibility. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for clipping. |
| `NumDecisionSteps` | Gets or sets the number of sequential decision steps. |
| `NumSharedLayers` | Gets or sets the number of shared layers in the feature transformer. |
| `NumStepSpecificLayers` | Gets or sets the number of decision-step-specific layers. |
| `OutputDimension` | Gets or sets the dimension of the output at each decision step. |
| `PreTrainingMaskingRatio` | Gets or sets the masking ratio for pre-training. |
| `RelaxationFactor` | Gets or sets the relaxation factor (gamma) for attention mask reuse. |
| `SparsityCoefficient` | Gets or sets the sparsity coefficient for the attention masks. |
| `VirtualBatchSize` | Gets or sets the virtual batch size for Ghost Batch Normalization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a copy of the options. |

