---
title: "SmartExplainerSelector<T>"
description: "Automatically selects the optimal explainer based on model type and provides caching for batch explanations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Automatically selects the optimal explainer based on model type and provides caching for batch explanations.

## For Beginners

Different explanation methods work best for different model types:

**Model Type → Best Explainer:**

- Decision Tree → TreeSHAP (exact, fast)
- Random Forest → TreeSHAP (exact, fast)
- Gradient Boosting → TreeSHAP (exact, fast)
- Neural Network → DeepSHAP or Integrated Gradients (uses gradients)
- Any Model → KernelSHAP or LIME (model-agnostic, slower)

**Why auto-selection matters:**

- TreeSHAP is O(TLD²) exact for trees but doesn't work on neural networks
- DeepSHAP is fast for neural networks but needs gradient access
- KernelSHAP works everywhere but is O(2^M) approximation

This class automatically picks the best method for your model type.

**Caching:**
Computing explanations is expensive. This class caches explanations so repeated
requests for the same input return instantly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SmartExplainerSelector(Func<Matrix<>,Vector<>>,Int32,ExplainableModelType,String[],Matrix<>,SmartExplainerOptions)` | Initializes a smart explainer selector with a prediction function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheHitRate` | Gets the cache hit rate. |
| `DetectedModelType` | Gets the detected model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache` | Clears the explanation cache. |
| `ComputeCacheKey(Vector<>,Nullable<Int32>)` | Computes cache key for an instance. |
| `ComputeExplanation(Vector<>,Nullable<Int32>,Nullable<ExplainerType>)` | Computes explanation using the appropriate method. |
| `CreateDefaultBackground(Int32)` | Creates default background data (zeros). |
| `CreateDefaultFeatureNames` | Creates default feature names. |
| `Explain(Vector<>,Nullable<Int32>)` | Explains a single instance using the automatically selected method. |
| `ExplainBatch(Matrix<>,Nullable<Int32>)` | Explains multiple instances in batch. |
| `ExplainWith(Vector<>,ExplainerType,Nullable<Int32>)` | Forces use of a specific explainer type. |
| `ExplainWithIntegratedGradients(Vector<>,Nullable<Int32>)` | Explains with Integrated Gradients. |
| `ExplainWithLIME(Vector<>)` | Explains with LIME. |
| `ExplainWithSHAP(Vector<>)` | Explains with Kernel SHAP. |
| `GetRecommendedExplainer` | Gets the recommended explainer type for the detected model. |

