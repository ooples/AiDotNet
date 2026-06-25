---
title: "InterpretableModelHelper"
description: "Provides static helper methods for model interpretability operations using production-ready explainer algorithms."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability`

Provides static helper methods for model interpretability operations using production-ready explainer algorithms.

## For Beginners

This class serves as a facade that connects interpretable models to the
actual explainer implementations (SHAP, LIME, PDP, Integrated Gradients, DeepLIFT, GradCAM, etc.).
It handles the conversion between model interfaces and the prediction functions that explainers need.

## How It Works

**IMPORTANT:** These methods delegate to production-ready explainer classes:

- SHAPExplainer: Kernel SHAP for model-agnostic Shapley value explanations
- LIMEExplainer: Local interpretable model-agnostic explanations
- PartialDependenceExplainer: PDP and ICE curves
- FeatureInteractionExplainer: Friedman's H-statistic
- IntegratedGradientsExplainer: Neural network attribution (requires gradient access)
- DeepLIFTExplainer: Fast neural network attribution
- GradCAMExplainer: Visual CNN explanations
- PermutationFeatureImportance: Model-agnostic global importance

**CRITICAL:** Methods that require background data will throw if no data is provided.
Synthetic background data is only used as a last resort and will emit warnings.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertTensorToMatrix(Tensor<>)` | Converts a Tensor to a Matrix (assuming 2D or flattening to 2D). |
| `CreateBackgroundData(IInterpretableModel<>,Int32,Int32)` | Creates synthetic background data for explainers (used as fallback). |
| `CreatePredictionFunction(IInterpretableModel<>)` | Creates a Matrix-based prediction function from an interpretable model. |
| `CreateTensorPredictionFunction(IInterpretableModel<>)` | Creates a Tensor-based prediction function from an interpretable model. |
| `CreateVectorPredictionFunction(IInterpretableModel<>)` | Creates a Vector-based prediction function from an interpretable model. |
| `GenerateTextExplanationAsync(IInterpretableModel<>,Tensor<>,Tensor<>)` | Generates a text explanation for a prediction. |
| `GetAnchorExplanationAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>,)` | Gets anchor explanation for a given input using beam search to find sufficient conditions (rules) that anchor the prediction. |
| `GetCounterfactualAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>,Tensor<>,Int32)` | Gets counterfactual explanation for a given input and desired output. |
| `GetDeepLIFTAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>,Tensor<>,DeepLIFTRule)` | Gets DeepLIFT attributions for neural network explanation. |
| `GetDeepLIFTWithBackpropAsync(INeuralNetwork<>,HashSet<InterpretationMethod>,Tensor<>,Tensor<>,DeepLIFTRule,String[])` | Gets DeepLIFT attributions using efficient backpropagation for a neural network. |
| `GetFeatureInteractionAsync(HashSet<InterpretationMethod>,Int32,Int32)` | Legacy overload without model parameter. |
| `GetFeatureInteractionAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Int32,Int32)` | Gets feature interaction (backwards compatible overload). |
| `GetFeatureInteractionAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Int32,Int32,Matrix<>,Int32)` | Gets feature interaction effects between two features using Friedman's H-statistic. |
| `GetGlobalFeatureImportanceAsync(IInterpretableModel<>,HashSet<InterpretationMethod>)` | Gets the global feature importance (simplified overload for backwards compatibility). |
| `GetGlobalFeatureImportanceAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Matrix<>,Vector<>,Int32)` | Gets the global feature importance using permutation feature importance. |
| `GetGradCAMAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>,Int32[],Int32[],Int32,Func<Tensor<>,Int32,Tensor<>>,Func<Tensor<>,Int32,Int32,Tensor<>>,Boolean)` | Gets Grad-CAM visual explanation for a CNN prediction. |
| `GetIntegratedGradientsAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>,Tensor<>,Int32,Func<Vector<>,Int32,Vector<>>)` | Gets Integrated Gradients attributions for neural network explanation. |
| `GetIntegratedGradientsWithBackpropAsync(INeuralNetwork<>,HashSet<InterpretationMethod>,Tensor<>,Tensor<>,Int32,String[])` | Gets Integrated Gradients attributions using efficient backpropagation for a neural network. |
| `GetLimeExplanationAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>,Int32,Int32,Double)` | Gets LIME explanation for a specific input using local linear approximation. |
| `GetLocalFeatureImportanceAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>)` | Gets the local feature importance for a specific input using gradient-based attribution. |
| `GetModelSpecificInterpretabilityAsync(IInterpretableModel<>)` | Gets model-specific interpretability information. |
| `GetNumFeatures(IInterpretableModel<>)` | Gets the number of input features from a model. |
| `GetPartialDependenceAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Vector<Int32>,Int32)` | Gets partial dependence data (backwards compatible overload with synthetic data fallback). |
| `GetPartialDependenceAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Vector<Int32>,Matrix<>,Int32,Boolean)` | Gets partial dependence data for specified features. |
| `GetShapValuesAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>)` | Gets SHAP values (backwards compatible overload that uses input as background). |
| `GetShapValuesAsync(IInterpretableModel<>,HashSet<InterpretationMethod>,Tensor<>,Matrix<>,Int32)` | Gets SHAP values for the given inputs using Kernel SHAP algorithm. |
| `GetTreeSHAPAsync(DecisionTreeNode<>,HashSet<InterpretationMethod>,Tensor<>,,String[])` | Gets TreeSHAP values for a tree-based model. |
| `GetTreeSHAPAsync(IEnumerable<DecisionTreeNode<>>,HashSet<InterpretationMethod>,Tensor<>,,String[])` | Gets TreeSHAP values for an ensemble of trees (Random Forest, Gradient Boosting). |
| `ValidateFairnessAsync(IInterpretableModel<>,Tensor<>,Int32,List<FairnessMetric>)` | Validates fairness metrics without ground truth (uses approximations). |
| `ValidateFairnessAsync(IInterpretableModel<>,Tensor<>,Vector<>,Int32,List<FairnessMetric>)` | Validates fairness metrics for the given inputs with ground truth labels. |
| `ValidateFairnessAsync(List<FairnessMetric>)` | Legacy overload without input data. |

