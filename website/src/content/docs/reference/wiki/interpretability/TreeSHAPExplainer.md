---
title: "TreeSHAPExplainer<T>"
description: "TreeSHAP explainer for computing exact SHAP values for tree-based models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

TreeSHAP explainer for computing exact SHAP values for tree-based models.
Implements the exact O(TLD²) algorithm from the Lundberg paper.

## For Beginners

TreeSHAP is a fast, exact algorithm for computing SHAP values
specifically designed for tree-based models (decision trees, random forests, gradient boosting).

Unlike Kernel SHAP which approximates SHAP values through sampling, TreeSHAP computes
the exact Shapley values by efficiently traversing the tree structure.

Key advantages over Kernel SHAP:

- **Exact values**: No approximation, mathematically precise results
- **Fast**: O(TLD²) complexity where T=trees, L=leaves, D=depth
- **No background data needed**: Uses the tree structure itself

This implementation follows the algorithm from:
Lundberg, Lee, et al. "Consistent Individualized Feature Attribution for Tree Ensembles"
arXiv:1802.03888 (2018)

TreeSHAP satisfies important properties:

- **Local accuracy**: SHAP values sum to (prediction - expected_prediction)
- **Consistency**: If a feature's contribution increases, its SHAP value increases
- **Missingness**: Missing features get zero attribution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TreeSHAPExplainer(DecisionTreeNode<>,Int32,,String[])` | Initializes a new TreeSHAP explainer for a single decision tree. |
| `TreeSHAPExplainer(IEnumerable<DecisionTreeNode<>>,Int32,,String[])` | Initializes a new TreeSHAP explainer for an ensemble of trees. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeTreeSHAP(DecisionTreeNode<>,Vector<>)` | Computes exact TreeSHAP values for a single tree using the Lundberg O(TLD²) algorithm. |
| `Explain(Vector<>)` | Computes TreeSHAP values for an input instance. |
| `ExplainBatch(Matrix<>)` |  |
| `ExtendPath(Int32)` | Extends the path by updating permutation weights. |
| `FindFeatureInPath(Int32,Int32)` | Finds a feature in the current path. |
| `PredictInstance(Vector<>)` | Predicts the output for an instance using the tree(s). |
| `PredictTree(DecisionTreeNode<>,Vector<>)` | Predicts the output for a single tree. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `TreeSHAPRecursive(DecisionTreeNode<>,Vector<>,Double[],Int32)` | Recursive TreeSHAP computation following the exact Lundberg algorithm. |
| `UnwindPath(Double[],Int32,Double)` | Unwinds the path at a leaf to compute SHAP contributions. |

