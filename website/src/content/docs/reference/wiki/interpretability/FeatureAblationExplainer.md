---
title: "FeatureAblationExplainer<T>"
description: "Feature Ablation explainer for understanding feature importance by removal."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Feature Ablation explainer for understanding feature importance by removal.

## For Beginners

Feature Ablation is a simple way to understand feature importance.
The idea is: replace each feature with a "baseline" value and see how the prediction changes.

**How it works:**

1. For each feature (or group of features):
2. Replace the feature with a baseline (e.g., zero, mean, or reference value)
3. Measure how much the prediction changes
4. Features that cause large changes are important

**Difference from Occlusion:**

- Occlusion: Slides a window over spatial data (images)
- Feature Ablation: Works on individual features (tabular data, any modality)

**Key advantage:** You can group features and ablate them together:

- All color channels for an image
- All words in a sentence segment
- All related features in tabular data

**Use cases:**

- Understanding which features drive predictions in tabular data
- Grouping related features (e.g., all "income" related columns)
- Debugging models by finding unexpected important features

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureAblationExplainer(Func<Tensor<>,Tensor<>>,Int32[][],String[],Boolean)` | Initializes a Feature Ablation explainer for tensor inputs. |
| `FeatureAblationExplainer(Func<Vector<>,Vector<>>,Vector<>,Int32[][],String[],Boolean)` | Initializes a new Feature Ablation explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` | Gets the method name. |
| `SupportsGlobalExplanations` | Gets whether this explainer supports global explanations. |
| `SupportsLocalExplanations` | Gets whether this explainer supports local explanations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AblateFeatures(Vector<>,Vector<>,Int32[])` | Ablates specified features in the input. |
| `CreateDefaultGroups(Int32)` | Creates default feature groups (one per feature). |
| `Explain(Vector<>)` | Explains a single input (ILocalExplainer interface). |
| `ExplainBatch(Matrix<>)` | Explains a batch of inputs (ILocalExplainer interface). |
| `ExplainGlobal(Matrix<>)` | Computes global feature importance across a dataset. |
| `ExplainGlobal(Matrix<>,Nullable<Int32>)` | Computes global feature importance across a dataset for a specific class. |
| `ExplainLocal(Vector<>,Nullable<Int32>)` | Explains a single input by ablating features. |
| `GetPredictedClass(Vector<>)` | Gets the predicted class from output. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

