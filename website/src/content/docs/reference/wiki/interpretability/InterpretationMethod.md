---
title: "InterpretationMethod"
description: "Enumeration of interpretation methods supported by interpretable models."
section: "API Reference"
---

`Enums` · `AiDotNet.Interpretability`

Enumeration of interpretation methods supported by interpretable models.

## For Beginners

These are the different explanation techniques you can use to understand
why your model makes certain predictions. Each method has different strengths:

- **Model-Agnostic Methods** (work with any model):
- SHAP, LIME, PartialDependence, Counterfactual, DiCE, Anchor, FeatureImportance, FeatureInteraction, Occlusion, FeatureAblation

- **Neural Network Methods** (require gradient access):
- IntegratedGradients, DeepLIFT, DeepSHAP, GradientSHAP, GradCAM, LayerGradCAM, GuidedBackprop, GuidedGradCAM, NoiseTunnel

- **Tree-Based Methods** (for tree models):
- TreeSHAP

- **Concept-Based Methods** (for high-level understanding):
- TCAV

- **Training Data Attribution**:
- InfluenceFunctions

Enable methods through `InterpretabilityOptions` when configuring your model.

## Fields

| Field | Summary |
|:-----|:--------|
| `Anchor` | Anchor explanations for rule-based interpretations. |
| `Counterfactual` | Counterfactual explanations to understand decision boundaries. |
| `DeepLIFT` | DeepLIFT (Deep Learning Important FeaTures) for neural network attribution. |
| `DeepSHAP` | DeepSHAP combining GradientSHAP with DeepLIFT for efficient neural network explanations. |
| `DiCE` | DiCE (Diverse Counterfactual Explanations) for generating diverse what-if scenarios. |
| `FeatureAblation` | Feature Ablation for attribution by removing/replacing features. |
| `FeatureImportance` | Feature importance analysis using permutation importance. |
| `FeatureInteraction` | Feature interaction analysis using Friedman's H-statistic. |
| `GradCAM` | Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN visual explanations. |
| `GradientSHAP` | GradientSHAP for gradient-based SHAP approximation on neural networks. |
| `GuidedBackprop` | Guided Backpropagation for cleaner gradient visualizations. |
| `GuidedGradCAM` | GuidedGradCAM combining GuidedBackprop with GradCAM for high-resolution class-specific visualizations. |
| `InfluenceFunctions` | Influence Functions for training data attribution. |
| `InputXGradient` | Input × Gradient attribution - multiplies input values by their gradients. |
| `IntegratedGradients` | Integrated Gradients for neural network attribution. |
| `LIME` | LIME (Local Interpretable Model-agnostic Explanations) for local explanations. |
| `LayerAttribution` | Layer-level attribution for computing attributions at intermediate layers. |
| `LayerGradCAM` | LayerGradCAM for class activation mapping at a specific network layer. |
| `NeuronAttribution` | Neuron-level attribution for understanding individual neuron contributions. |
| `NoiseTunnel` | NoiseTunnel (SmoothGrad) for noise-averaged gradient smoothing. |
| `Occlusion` | Occlusion-based attribution by systematically hiding parts of input. |
| `PartialDependence` | Partial dependence plots to show feature effects. |
| `SHAP` | SHAP (SHapley Additive exPlanations) values for feature importance. |
| `TCAV` | TCAV (Testing with Concept Activation Vectors) for concept-level explanations. |
| `TracIn` | TracIn (Tracing Influence) for efficient training data attribution. |
| `TreeSHAP` | TreeSHAP for exact SHAP values on tree-based models. |

