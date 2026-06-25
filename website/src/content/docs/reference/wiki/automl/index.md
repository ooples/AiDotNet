---
title: "AutoML"
description: "All 45 public types in the AiDotNet.automl namespace, organized by kind."
section: "API Reference"
---

**45** public types in this namespace, organized by kind.

## Models & Types (31)

| Type | Summary |
|:-----|:--------|
| [`ArchitectureDto`](/docs/reference/wiki/automl/architecturedto/) | Data transfer object for architecture JSON serialization. |
| [`Architecture<T>`](/docs/reference/wiki/automl/architecture/) | Represents a neural network architecture discovered through NAS. |
| [`AttentiveNAS<T>`](/docs/reference/wiki/automl/attentivenas/) | AttentiveNAS: Improving Neural Architecture Search via Attentive Sampling. |
| [`AutoMLEnsembleModel<T>`](/docs/reference/wiki/automl/automlensemblemodel/) | A simple tabular ensemble model used as a facade-safe AutoML final model. |
| [`BayesianOptimizationAutoML<T, TInput, TOutput>`](/docs/reference/wiki/automl/bayesianoptimizationautoml/) | Built-in AutoML strategy that uses a lightweight Bayesian-style surrogate to guide trial selection. |
| [`BigNAS<T>`](/docs/reference/wiki/automl/bignas/) | BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models. |
| [`CompressionOptimizer<T>`](/docs/reference/wiki/automl/compressionoptimizer/) | Automatically finds the best compression configuration for a model. |
| [`CompressionTrial<T>`](/docs/reference/wiki/automl/compressiontrial/) | Represents a compression configuration to be evaluated. |
| [`DiffusionAutoML<T>`](/docs/reference/wiki/automl/diffusionautoml/) | AutoML for diffusion models with automatic hyperparameter optimization. |
| [`ENAS<T>`](/docs/reference/wiki/automl/enas/) | Efficient Neural Architecture Search via Parameter Sharing. |
| [`EvolutionaryAutoML<T, TInput, TOutput>`](/docs/reference/wiki/automl/evolutionaryautoml/) | Built-in AutoML strategy that uses an evolutionary (genetic) approach to propose new trials. |
| [`FBNet<T>`](/docs/reference/wiki/automl/fbnet/) | FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search. |
| [`GDAS<T>`](/docs/reference/wiki/automl/gdas/) | Gradient-based Differentiable Architecture Search with Gumbel-Softmax sampling. |
| [`HardwareConstraints<T>`](/docs/reference/wiki/automl/hardwareconstraints/) | Hardware constraints for NAS. |
| [`HardwareCostModel<T>`](/docs/reference/wiki/automl/hardwarecostmodel/) | Models hardware costs for neural architecture search operations using FLOP-based estimation. |
| [`HardwareCost<T>`](/docs/reference/wiki/automl/hardwarecost/) | Represents the hardware cost of an operation or architecture. |
| [`MobileNetSearchSpace<T>`](/docs/reference/wiki/automl/mobilenetsearchspace/) | Defines the MobileNet-based search space for neural architecture search. |
| [`MultiFidelityAutoML<T, TInput, TOutput>`](/docs/reference/wiki/automl/multifidelityautoml/) | Built-in AutoML strategy that uses multi-fidelity (successive halving) and ASHA scheduling. |
| [`NeuralArchitectureSearch<T>`](/docs/reference/wiki/automl/neuralarchitecturesearch/) | Neural Architecture Search implementation with gradient-based (DARTS) support |
| [`OnceForAll<T>`](/docs/reference/wiki/automl/onceforall/) | Once-for-All (OFA) Networks: Train Once, Specialize for Anything. |
| [`OperationDto`](/docs/reference/wiki/automl/operationdto/) | Data transfer object for a single operation in the architecture. |
| [`PCDARTS<T>`](/docs/reference/wiki/automl/pcdarts/) | Partial Channel Connections for Memory-Efficient Differentiable Architecture Search. |
| [`ParameterRange`](/docs/reference/wiki/automl/parameterrange/) | Defines the range and type of a hyperparameter for AutoML search |
| [`PlatformCharacteristics`](/docs/reference/wiki/automl/platformcharacteristics/) | Platform characteristics for hardware cost estimation. |
| [`ProxylessNAS<T>`](/docs/reference/wiki/automl/proxylessnas/) | ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. |
| [`RandomSearchAutoML<T, TInput, TOutput>`](/docs/reference/wiki/automl/randomsearchautoml/) | AutoML implementation that uses random search over candidate model types and hyperparameters. |
| [`ResNetSearchSpace<T>`](/docs/reference/wiki/automl/resnetsearchspace/) | Defines the ResNet-based search space for neural architecture search. |
| [`SearchConstraint`](/docs/reference/wiki/automl/searchconstraint/) | Defines a constraint for AutoML search to limit the search space or enforce requirements. |
| [`SearchSpaceBase<T>`](/docs/reference/wiki/automl/searchspacebase/) | Defines the search space for neural architecture search. |
| [`TransformerSearchSpace<T>`](/docs/reference/wiki/automl/transformersearchspace/) | Defines the Transformer-based search space for neural architecture search. |
| [`TrialResult`](/docs/reference/wiki/automl/trialresult/) | Represents the result of a single trial during AutoML search. |

## Base Classes (5)

| Type | Summary |
|:-----|:--------|
| [`AutoMLModelBase<T, TInput, TOutput>`](/docs/reference/wiki/automl/automlmodelbase/) | Base class for AutoML models that automatically search for optimal model configurations |
| [`BuiltInSupervisedAutoMLModelBase<T, TInput, TOutput>`](/docs/reference/wiki/automl/builtinsupervisedautomlmodelbase/) | Base class for built-in supervised AutoML strategies that operate on tabular Matrix/Vector tasks. |
| [`NasAutoMLModelBase<T>`](/docs/reference/wiki/automl/nasautomlmodelbase/) | Base class for NAS-based AutoML models. |
| [`SupervisedAutoMLModelBase<T, TInput, TOutput>`](/docs/reference/wiki/automl/supervisedautomlmodelbase/) | Base class for AutoML implementations that train and score supervised models. |
| [`UnsupervisedAutoMLBase<T>`](/docs/reference/wiki/automl/unsupervisedautomlbase/) | Base class for unsupervised AutoML search strategies (e.g., clustering AutoML, grid search). |

## Enums (4)

| Type | Summary |
|:-----|:--------|
| [`ConstraintType`](/docs/reference/wiki/automl/constrainttype/) | Defines the types of constraints that can be applied to AutoML search. |
| [`DiffusionSchedulerType`](/docs/reference/wiki/automl/diffusionschedulertype/) | Represents the type of scheduler for diffusion sampling. |
| [`HardwarePlatform`](/docs/reference/wiki/automl/hardwareplatform/) | Supported hardware platforms. |
| [`NoisePredictorType`](/docs/reference/wiki/automl/noisepredictortype/) | Represents the type of noise predictor architecture. |

## Options & Configuration (5)

| Type | Summary |
|:-----|:--------|
| [`AttentiveNASConfig<T>`](/docs/reference/wiki/automl/attentivenasconfig/) | Configuration for an AttentiveNAS sub-network. |
| [`BigNASConfig`](/docs/reference/wiki/automl/bignasconfig/) | Configuration for a BigNAS sub-network. |
| [`CompressionOptimizerOptions`](/docs/reference/wiki/automl/compressionoptimizeroptions/) | Configuration options for the compression optimizer. |
| [`DiffusionTrialConfig<T>`](/docs/reference/wiki/automl/diffusiontrialconfig/) | Configuration for a diffusion model trial in AutoML. |
| [`SubNetworkConfig`](/docs/reference/wiki/automl/subnetworkconfig/) | Configuration for a sub-network sampled from OFA. |

