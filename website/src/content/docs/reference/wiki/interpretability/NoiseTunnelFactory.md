---
title: "NoiseTunnelFactory"
description: "Convenience factory for creating NoiseTunnel wrappers for common explainer types."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability.Explainers`

Convenience factory for creating NoiseTunnel wrappers for common explainer types.

## Methods

| Method | Summary |
|:-----|:--------|
| `ForGuidedBackprop(ILocalExplainer<,GuidedBackpropExplanation<>>,NoiseTunnelType,Int32,Double,Nullable<Int32>)` | Creates a NoiseTunnel wrapper for GuidedBackprop explainers. |
| `ForIntegratedGradients(ILocalExplainer<,IntegratedGradientsExplanation<>>,NoiseTunnelType,Int32,Double,Nullable<Int32>)` | Creates a NoiseTunnel wrapper for IntegratedGradients explainers. |
| `ForShap(ILocalExplainer<,SHAPExplanation<>>,NoiseTunnelType,Int32,Double,Nullable<Int32>)` | Creates a NoiseTunnel wrapper for SHAP explainers. |

