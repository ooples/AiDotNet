---
title: "Interpretability"
description: "All 118 public types in the AiDotNet.interpretability namespace, organized by kind."
section: "API Reference"
---

**118** public types in this namespace, organized by kind.

## Models & Types (90)

| Type | Summary |
|:-----|:--------|
| [`ALE2DResult<T>`](/docs/reference/wiki/interpretability/ale2dresult/) | Represents the result of a 2D ALE analysis (feature interaction). |
| [`ALEResult<T>`](/docs/reference/wiki/interpretability/aleresult/) | Represents the result of an ALE (Accumulated Local Effects) analysis. |
| [`AccumulatedLocalEffectsExplainer<T>`](/docs/reference/wiki/interpretability/accumulatedlocaleffectsexplainer/) | Model-agnostic Accumulated Local Effects (ALE) explainer. |
| [`AnchorExplainer<T>`](/docs/reference/wiki/interpretability/anchorexplainer/) | Model-agnostic Anchor explainer that provides rule-based explanations. |
| [`AnchorExplanation<T>`](/docs/reference/wiki/interpretability/anchorexplanation/) | Represents an anchor explanation providing rule-based interpretations. |
| [`AttentionExplanation<T>`](/docs/reference/wiki/interpretability/attentionexplanation/) | Represents the result of an Attention Visualization analysis. |
| [`AttentionVisualizationExplainer<T>`](/docs/reference/wiki/interpretability/attentionvisualizationexplainer/) | Attention Visualization explainer for Transformer models. |
| [`BackgroundSummary<T>`](/docs/reference/wiki/interpretability/backgroundsummary/) | Represents summarized background data for interpretability methods. |
| [`BasicFairnessEvaluator<T>`](/docs/reference/wiki/interpretability/basicfairnessevaluator/) | Basic fairness evaluator that computes only fundamental fairness metrics. |
| [`BiasDetectionResult<T>`](/docs/reference/wiki/interpretability/biasdetectionresult/) | Represents the results of a bias detection analysis. |
| [`ComprehensiveFairnessEvaluator<T>`](/docs/reference/wiki/interpretability/comprehensivefairnessevaluator/) | Comprehensive fairness evaluator that computes all major fairness metrics. |
| [`ConceptActivationVector<T>`](/docs/reference/wiki/interpretability/conceptactivationvector/) | Represents a Concept Activation Vector. |
| [`ContrastiveExplainer<T>`](/docs/reference/wiki/interpretability/contrastiveexplainer/) | Contrastive explainer that answers "Why X and not Y?" questions. |
| [`ContrastiveExplanation<T>`](/docs/reference/wiki/interpretability/contrastiveexplanation/) | Represents the result of a Contrastive explanation. |
| [`CounterfactualExplainer<T>`](/docs/reference/wiki/interpretability/counterfactualexplainer/) | Model-agnostic Counterfactual explainer that finds minimal changes needed for a different prediction. |
| [`CounterfactualExplanation<T>`](/docs/reference/wiki/interpretability/counterfactualexplanation/) | Represents a counterfactual explanation showing minimal changes needed for a different outcome. |
| [`DeepLIFTExplainer<T>`](/docs/reference/wiki/interpretability/deepliftexplainer/) | DeepLIFT (Deep Learning Important FeaTures) explainer for neural networks. |
| [`DeepLIFTExplanation<T>`](/docs/reference/wiki/interpretability/deepliftexplanation/) | Represents the result of a DeepLIFT analysis. |
| [`DeepSHAPExplainer<T>`](/docs/reference/wiki/interpretability/deepshapexplainer/) | DeepSHAP explainer combining GradientSHAP with DeepLIFT for efficient neural network explanations. |
| [`DeepSHAPExplanation<T>`](/docs/reference/wiki/interpretability/deepshapexplanation/) | Represents the result of a DeepSHAP analysis. |
| [`DemographicParityBiasDetector<T>`](/docs/reference/wiki/interpretability/demographicparitybiasdetector/) | Detects bias using Demographic Parity (Statistical Parity Difference). |
| [`DiCEExplainer<T>`](/docs/reference/wiki/interpretability/diceexplainer/) | DiCE (Diverse Counterfactual Explanations) explainer using genetic algorithm-based search. |
| [`DiCEExplanation<T>`](/docs/reference/wiki/interpretability/diceexplanation/) | DiCE explanation containing multiple diverse counterfactuals. |
| [`DisparateImpactBiasDetector<T>`](/docs/reference/wiki/interpretability/disparateimpactbiasdetector/) | Detects bias using the Disparate Impact metric (80% rule). |
| [`EqualOpportunityBiasDetector<T>`](/docs/reference/wiki/interpretability/equalopportunitybiasdetector/) | Detects bias using Equal Opportunity metric (True Positive Rate difference). |
| [`FairnessMetrics<T>`](/docs/reference/wiki/interpretability/fairnessmetrics/) | Represents fairness metrics for model evaluation. |
| [`FeatureAblationExplainer<T>`](/docs/reference/wiki/interpretability/featureablationexplainer/) | Feature Ablation explainer for understanding feature importance by removal. |
| [`FeatureAblationExplanation<T>`](/docs/reference/wiki/interpretability/featureablationexplanation/) | Result of feature ablation for a single input. |
| [`FeatureAttribution<T>`](/docs/reference/wiki/interpretability/featureattribution/) | Generic feature attribution result. |
| [`FeatureChange<T>`](/docs/reference/wiki/interpretability/featurechange/) | Represents a single feature change in a counterfactual. |
| [`FeatureImportanceResult<T>`](/docs/reference/wiki/interpretability/featureimportanceresult/) | Represents the result of permutation feature importance calculation. |
| [`FeatureInteractionExplainer<T>`](/docs/reference/wiki/interpretability/featureinteractionexplainer/) | Model-agnostic Feature Interaction detector using Friedman's H-statistic. |
| [`FeatureInteractionResult<T>`](/docs/reference/wiki/interpretability/featureinteractionresult/) | Represents the result of a Feature Interaction analysis. |
| [`GlobalDeepSHAPExplanation<T>`](/docs/reference/wiki/interpretability/globaldeepshapexplanation/) | Represents global DeepSHAP feature importance. |
| [`GlobalFeatureAblationResult<T>`](/docs/reference/wiki/interpretability/globalfeatureablationresult/) | Global feature ablation result across a dataset. |
| [`GlobalSHAPExplanation<T>`](/docs/reference/wiki/interpretability/globalshapexplanation/) | Represents global SHAP explanations aggregated across multiple instances. |
| [`GlobalSurrogateExplainer<T>`](/docs/reference/wiki/interpretability/globalsurrogateexplainer/) | Global Surrogate Model explainer that approximates a complex model with an interpretable one. |
| [`GradCAMExplainer<T>`](/docs/reference/wiki/interpretability/gradcamexplainer/) | Gradient-weighted Class Activation Mapping (Grad-CAM) explainer for CNNs. |
| [`GradCAMExplanation<T>`](/docs/reference/wiki/interpretability/gradcamexplanation/) | Represents the result of a Grad-CAM analysis. |
| [`GradientSHAPExplainer<T>`](/docs/reference/wiki/interpretability/gradientshapexplainer/) | GradientSHAP explainer - a faster approximation of SHAP using gradients. |
| [`GradientSHAPExplanation<T>`](/docs/reference/wiki/interpretability/gradientshapexplanation/) | Represents the result of a GradientSHAP analysis. |
| [`GroupFairnessEvaluator<T>`](/docs/reference/wiki/interpretability/groupfairnessevaluator/) | Group-level fairness evaluator that focuses on equalized performance across groups. |
| [`GuidedBackpropExplainer<T>`](/docs/reference/wiki/interpretability/guidedbackpropexplainer/) | Guided Backpropagation explainer for neural network visualization. |
| [`GuidedBackpropExplanation<T>`](/docs/reference/wiki/interpretability/guidedbackpropexplanation/) | Result of Guided Backpropagation explanation. |
| [`GuidedGradCAMExplainer<T>`](/docs/reference/wiki/interpretability/guidedgradcamexplainer/) | Guided GradCAM explainer combining GuidedBackprop with GradCAM for high-resolution explanations. |
| [`GuidedGradCAMExplanation<T>`](/docs/reference/wiki/interpretability/guidedgradcamexplanation/) | Guided GradCAM explanation result. |
| [`InfluenceFunctionExplainer<T>`](/docs/reference/wiki/interpretability/influencefunctionexplainer/) | Influence Function explainer for training data attribution. |
| [`InfluenceFunctionResult<T>`](/docs/reference/wiki/interpretability/influencefunctionresult/) | Result of influence function computation. |
| [`InputXGradientExplainer<T>`](/docs/reference/wiki/interpretability/inputxgradientexplainer/) | Input × Gradient attribution explainer - multiplies input values by their gradients. |
| [`InputXGradientExplanation<T>`](/docs/reference/wiki/interpretability/inputxgradientexplanation/) | Result of Input × Gradient attribution. |
| [`IntegratedGradientsExplainer<T>`](/docs/reference/wiki/interpretability/integratedgradientsexplainer/) | Integrated Gradients explainer for neural networks with gradient access. |
| [`IntegratedGradientsExplanation<T>`](/docs/reference/wiki/interpretability/integratedgradientsexplanation/) | Represents the result of an Integrated Gradients analysis. |
| [`LIMEExplainer<T>`](/docs/reference/wiki/interpretability/limeexplainer/) | Model-agnostic LIME (Local Interpretable Model-agnostic Explanations) explainer. |
| [`LIMEExplanationResult<T>`](/docs/reference/wiki/interpretability/limeexplanationresult/) | Represents a LIME explanation result. |
| [`LRPExplanation<T>`](/docs/reference/wiki/interpretability/lrpexplanation/) | Represents the result of an LRP analysis. |
| [`LayerAttributionExplainer<T>`](/docs/reference/wiki/interpretability/layerattributionexplainer/) | Layer-level attribution explainer for computing attributions at intermediate layers. |
| [`LayerAttributionResult<T>`](/docs/reference/wiki/interpretability/layerattributionresult/) | Result of layer attribution. |
| [`LayerGradCAMExplainer<T>`](/docs/reference/wiki/interpretability/layergradcamexplainer/) | Layer GradCAM (Gradient-weighted Class Activation Mapping) explainer. |
| [`LayerGradCAMExplanation<T>`](/docs/reference/wiki/interpretability/layergradcamexplanation/) | GradCAM explanation result. |
| [`LayerwiseRelevancePropagationExplainer<T>`](/docs/reference/wiki/interpretability/layerwiserelevancepropagationexplainer/) | Layer-wise Relevance Propagation (LRP) explainer for neural networks. |
| [`LimeExplanation<T>`](/docs/reference/wiki/interpretability/limeexplanation/) | Represents a LIME (Local Interpretable Model-agnostic Explanations) explanation for a prediction. |
| [`LocalTCAVExplanation<T>`](/docs/reference/wiki/interpretability/localtcavexplanation/) | Represents a local TCAV explanation for a single input. |
| [`NeuronAttributionExplainer<T>`](/docs/reference/wiki/interpretability/neuronattributionexplainer/) | Neuron-level attribution explainer for understanding individual neuron contributions. |
| [`NeuronAttributionResult<T>`](/docs/reference/wiki/interpretability/neuronattributionresult/) | Result of neuron attribution. |
| [`NoiseTunnelExplainer<T, TExplanation>`](/docs/reference/wiki/interpretability/noisetunnelexplainer/) | Noise Tunnel wrapper that smooths attributions by averaging over noisy inputs. |
| [`OcclusionExplainer<T>`](/docs/reference/wiki/interpretability/occlusionexplainer/) | Occlusion explainer for image and sequential data interpretation. |
| [`OcclusionExplanation<T>`](/docs/reference/wiki/interpretability/occlusionexplanation/) | Result of occlusion analysis. |
| [`PartialDependence2DResult<T>`](/docs/reference/wiki/interpretability/partialdependence2dresult/) | Represents the result of a 2D partial dependence analysis (feature interaction). |
| [`PartialDependenceData<T>`](/docs/reference/wiki/interpretability/partialdependencedata/) | Represents partial dependence data showing how features affect predictions. |
| [`PartialDependenceExplainer<T>`](/docs/reference/wiki/interpretability/partialdependenceexplainer/) | Model-agnostic Partial Dependence Plot (PDP) explainer with Individual Conditional Expectation (ICE) curves. |
| [`PartialDependenceResult<T>`](/docs/reference/wiki/interpretability/partialdependenceresult/) | Represents the result of a partial dependence analysis. |
| [`PermutationFeatureImportance<T>`](/docs/reference/wiki/interpretability/permutationfeatureimportance/) | Model-agnostic Permutation Feature Importance calculator. |
| [`PertinentFeature<T>`](/docs/reference/wiki/interpretability/pertinentfeature/) | Represents a pertinent feature in a contrastive explanation. |
| [`PrototypeExplainer<T>`](/docs/reference/wiki/interpretability/prototypeexplainer/) | Prototype-based explainer that explains predictions using similar examples. |
| [`PrototypeExplanation<T>`](/docs/reference/wiki/interpretability/prototypeexplanation/) | Represents the result of a Prototype-based explanation. |
| [`PrototypeMatch<T>`](/docs/reference/wiki/interpretability/prototypematch/) | Represents a matched prototype. |
| [`SHAPExplainer<T>`](/docs/reference/wiki/interpretability/shapexplainer/) | Model-agnostic SHAP (SHapley Additive exPlanations) explainer using Kernel SHAP algorithm. |
| [`SHAPExplanation<T>`](/docs/reference/wiki/interpretability/shapexplanation/) | Represents a SHAP explanation for a single prediction. |
| [`SaliencyMapExplainer<T>`](/docs/reference/wiki/interpretability/saliencymapexplainer/) | Saliency Map explainer using gradient-based methods. |
| [`SaliencyMapExplanation<T>`](/docs/reference/wiki/interpretability/saliencymapexplanation/) | Represents the result of a Saliency Map analysis. |
| [`SelfInfluenceResult<T>`](/docs/reference/wiki/interpretability/selfinfluenceresult/) | Result of self-influence computation. |
| [`SingleCounterfactual<T>`](/docs/reference/wiki/interpretability/singlecounterfactual/) | Represents a single counterfactual explanation. |
| [`SmartExplainerSelector<T>`](/docs/reference/wiki/interpretability/smartexplainerselector/) | Automatically selects the optimal explainer based on model type and provides caching for batch explanations. |
| [`SurrogateExplanation<T>`](/docs/reference/wiki/interpretability/surrogateexplanation/) | Represents a global surrogate model explanation. |
| [`TCAVExplainer<T>`](/docs/reference/wiki/interpretability/tcavexplainer/) | Testing with Concept Activation Vectors (TCAV) explainer. |
| [`TCAVResult<T>`](/docs/reference/wiki/interpretability/tcavresult/) | Represents the result of a TCAV analysis for a single concept. |
| [`TCAVResults<T>`](/docs/reference/wiki/interpretability/tcavresults/) | Represents TCAV results for multiple concepts. |
| [`TracInResult<T>`](/docs/reference/wiki/interpretability/tracinresult/) | Result of TracIn computation. |
| [`TreeSHAPExplainer<T>`](/docs/reference/wiki/interpretability/treeshapexplainer/) | TreeSHAP explainer for computing exact SHAP values for tree-based models. |
| [`TreeSHAPExplanation<T>`](/docs/reference/wiki/interpretability/treeshapexplanation/) | Represents the result of a TreeSHAP analysis. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`BiasDetectorBase<T>`](/docs/reference/wiki/interpretability/biasdetectorbase/) | Base class for all bias detectors that identify unfair treatment in model predictions. |
| [`FairnessEvaluatorBase<T>`](/docs/reference/wiki/interpretability/fairnessevaluatorbase/) | Base class for all fairness evaluators that measure equitable treatment in models. |

## Interfaces (3)

| Type | Summary |
|:-----|:--------|
| [`IConvolutionalNetwork<T, TInput, TOutput>`](/docs/reference/wiki/interpretability/iconvolutionalnetwork/) | Interface for convolutional neural networks that support Grad-CAM explanation. |
| [`IGPUAcceleratedExplainer<T>`](/docs/reference/wiki/interpretability/igpuacceleratedexplainer/) | Interface for explainers that support GPU acceleration. |
| [`ITransformerNetwork<T, TInput, TOutput>`](/docs/reference/wiki/interpretability/itransformernetwork/) | Interface for transformer networks that support attention visualization. |

## Enums (14)

| Type | Summary |
|:-----|:--------|
| [`DeepLIFTRule`](/docs/reference/wiki/interpretability/deepliftrule/) | DeepLIFT attribution rules. |
| [`DistanceMetric`](/docs/reference/wiki/interpretability/distancemetric/) | Distance metrics for prototype matching. |
| [`ExplainableModelType`](/docs/reference/wiki/interpretability/explainablemodeltype/) | Model types that the smart selector recognizes for choosing the optimal explainer. |
| [`ExplainerType`](/docs/reference/wiki/interpretability/explainertype/) | Explainer types available. |
| [`FairnessMetric`](/docs/reference/wiki/interpretability/fairnessmetric/) | Enumeration of fairness metrics for model evaluation. |
| [`FeatureType`](/docs/reference/wiki/interpretability/featuretype/) | Type of feature for counterfactual constraints. |
| [`InterpretationMethod`](/docs/reference/wiki/interpretability/interpretationmethod/) | Enumeration of interpretation methods supported by interpretable models. |
| [`InverseHessianMethod`](/docs/reference/wiki/interpretability/inversehessianmethod/) | Methods for computing inverse Hessian-vector products. |
| [`LRPRule`](/docs/reference/wiki/interpretability/lrprule/) | LRP propagation rules. |
| [`LayerAttributionMethod`](/docs/reference/wiki/interpretability/layerattributionmethod/) | Layer attribution methods. |
| [`NeuronAttributionMethod`](/docs/reference/wiki/interpretability/neuronattributionmethod/) | Neuron attribution methods. |
| [`NoiseTunnelType`](/docs/reference/wiki/interpretability/noisetunneltype/) | Types of noise tunnel aggregation methods. |
| [`OcclusionShape`](/docs/reference/wiki/interpretability/occlusionshape/) | Shape of occlusion patches. |
| [`SaliencyMethod`](/docs/reference/wiki/interpretability/saliencymethod/) | Saliency computation methods. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`SmartExplainerOptions`](/docs/reference/wiki/interpretability/smartexplaineroptions/) | Options for smart explainer selector. |

## Helpers & Utilities (8)

| Type | Summary |
|:-----|:--------|
| [`AutodiffGradientHelper<T>`](/docs/reference/wiki/interpretability/autodiffgradienthelper/) | Provides gradient computation using the GradientTape automatic differentiation system. |
| [`BackgroundSummarizer<T>`](/docs/reference/wiki/interpretability/backgroundsummarizer/) | Provides methods for summarizing background data for SHAP and other interpretability methods. |
| [`GPUExplainerHelper<T>`](/docs/reference/wiki/interpretability/gpuexplainerhelper/) | Provides GPU acceleration for interpretability explainers. |
| [`GradientHelperFactory<T>`](/docs/reference/wiki/interpretability/gradienthelperfactory/) | Factory methods for creating gradient helpers from various model types. |
| [`InputGradientHelper<T>`](/docs/reference/wiki/interpretability/inputgradienthelper/) | Provides unified input gradient computation for interpretability explainers. |
| [`InterpretabilityMetricsHelper<T>`](/docs/reference/wiki/interpretability/interpretabilitymetricshelper/) | Provides static utility methods for computing interpretability and fairness metrics. |
| [`InterpretableModelHelper`](/docs/reference/wiki/interpretability/interpretablemodelhelper/) | Provides static helper methods for model interpretability operations using production-ready explainer algorithms. |
| [`NoiseTunnelFactory`](/docs/reference/wiki/interpretability/noisetunnelfactory/) | Convenience factory for creating NoiseTunnel wrappers for common explainer types. |

