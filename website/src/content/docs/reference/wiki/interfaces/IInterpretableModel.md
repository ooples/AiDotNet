---
title: "IInterpretableModel<T>"
description: "Interface for models that support interpretability features."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for models that support interpretability features.

## For Beginners

This interface provides a unified way to explain model predictions.
Models implementing this interface can use various explanation techniques:

**Model-Agnostic Methods** (work with any model):

- SHAP: Shapley value-based feature attribution
- LIME: Local linear approximations
- Partial Dependence: Feature effect plots
- Counterfactual: "What-if" explanations
- Anchor: Rule-based explanations
- Feature Importance: Permutation-based importance
- Feature Interaction: H-statistic for interactions

**Neural Network Methods** (require gradient access):

- Integrated Gradients: Path-based attribution
- DeepLIFT: Activation-based attribution
- GradCAM: Visual CNN explanations

Enable methods through `InterpretationMethod[])` and configure via
`InterpretabilityOptions`.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigureFairness(Vector<Int32>,FairnessMetric[])` | Configures fairness evaluation settings. |
| `EnableMethod(InterpretationMethod[])` | Enables specific interpretation methods. |
| `GenerateTextExplanationAsync(Tensor<>,Tensor<>)` | Generates a text explanation for a prediction. |
| `GetAnchorExplanationAsync(Tensor<>,)` | Gets anchor explanation for a given input. |
| `GetCounterfactualAsync(Tensor<>,Tensor<>,Int32)` | Gets counterfactual explanation for a given input and desired output. |
| `GetDeepLIFTAsync(Tensor<>,Tensor<>,Boolean)` | Gets DeepLIFT attributions for a neural network prediction. |
| `GetFeatureInteractionAsync(Int32,Int32)` | Gets feature interaction effects between two features. |
| `GetGlobalFeatureImportanceAsync` | Gets the global feature importance across all predictions. |
| `GetGradCAMAsync(Tensor<>,Int32)` | Gets GradCAM visual explanation for a CNN prediction. |
| `GetIntegratedGradientsAsync(Tensor<>,Tensor<>,Int32)` | Gets Integrated Gradients attributions for a neural network prediction. |
| `GetLimeExplanationAsync(Tensor<>,Int32)` | Gets LIME explanation for a specific input. |
| `GetLocalFeatureImportanceAsync(Tensor<>)` | Gets the local feature importance for a specific input. |
| `GetModelSpecificInterpretabilityAsync` | Gets model-specific interpretability information. |
| `GetPartialDependenceAsync(Vector<Int32>,Int32)` | Gets partial dependence data for specified features. |
| `GetShapValuesAsync(Tensor<>)` | Gets SHAP values for the given inputs. |
| `SetBaseModel(IFullModel<,,>)` | Sets the base model for interpretability analysis. |
| `ValidateFairnessAsync(Tensor<>,Int32)` | Validates fairness metrics for the given inputs. |

