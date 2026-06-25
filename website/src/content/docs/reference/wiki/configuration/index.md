---
title: "Configuration"
description: "All 76 public types in the AiDotNet.configuration namespace, organized by kind."
section: "API Reference"
---

**76** public types in this namespace, organized by kind.

## Models & Types (14)

| Type | Summary |
|:-----|:--------|
| [`AiModelDataPipeline<T, TInput, TOutput>`](/docs/reference/wiki/configuration/aimodeldatapipeline/) | Default implementation of `IAiModelDataPipeline`. |
| [`AiModelTrainingCore<T, TInput, TOutput>`](/docs/reference/wiki/configuration/aimodeltrainingcore/) | Default implementation of `IAiModelTrainingCore`. |
| [`FusedOptimizerPathEvent`](/docs/reference/wiki/configuration/fusedoptimizerpathevent/) | Emitted when `TrainWithTape` enters or skips the fused-compiled fast path. |
| [`GradientNormEvent`](/docs/reference/wiki/configuration/gradientnormevent/) | Emitted per parameter tensor after `tape.ComputeGradients` returns, before the optimizer step runs. |
| [`RLEpisodeMetrics<T>`](/docs/reference/wiki/configuration/rlepisodemetrics/) | Metrics for a completed RL episode. |
| [`RLStepMetrics<T>`](/docs/reference/wiki/configuration/rlstepmetrics/) | Metrics for a single RL training step. |
| [`RLTrainingSummary<T>`](/docs/reference/wiki/configuration/rltrainingsummary/) | Summary of completed RL training. |
| [`TrainingLossEvent`](/docs/reference/wiki/configuration/traininglossevent/) | Emitted once per call to `TrainWithTape` with the scalar loss value returned by the loss function for that batch. |
| [`TrainingMessageEvent`](/docs/reference/wiki/configuration/trainingmessageevent/) | Free-form text event for callers that just want to emit a typed log line at a specific level. |
| [`YamlLicenseSection`](/docs/reference/wiki/configuration/yamllicensesection/) | YAML configuration section for license key settings. |
| [`YamlOptimizerSection`](/docs/reference/wiki/configuration/yamloptimizersection/) | YAML section for selecting an optimizer type by name. |
| [`YamlPipelineSection`](/docs/reference/wiki/configuration/yamlpipelinesection/) | YAML section for pipeline-style configurations (preprocessing, postprocessing, etc.). |
| [`YamlTimeSeriesModelSection`](/docs/reference/wiki/configuration/yamltimeseriesmodelsection/) | YAML section for selecting a time series model type by name. |
| [`YamlTypeSection`](/docs/reference/wiki/configuration/yamltypesection/) | Generic YAML section for any type-based configuration. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`TrainingDiagnosticEvent`](/docs/reference/wiki/configuration/trainingdiagnosticevent/) | Base type for structured training-pipeline diagnostic events. |

## Interfaces (2)

| Type | Summary |
|:-----|:--------|
| [`IAiModelDataPipeline<T, TInput, TOutput>`](/docs/reference/wiki/configuration/iaimodeldatapipeline/) | Component that owns the data-pipeline configuration for an AI model build: preprocessing, postprocessing, data loading, data preparation, and augmentation. |
| [`IAiModelTrainingCore<T, TInput, TOutput>`](/docs/reference/wiki/configuration/iaimodeltrainingcore/) | Component that owns the core training configuration for an AI model build: the model itself, optimizer, regularization, fitness calculator, fit detector, training pipeline, training monitor, checkpoint manager, and memory management. |

## Enums (12)

| Type | Summary |
|:-----|:--------|
| [`AttentionMaskingMode`](/docs/reference/wiki/configuration/attentionmaskingmode/) | Controls how attention masking is applied for optimized attention implementations. |
| [`CacheEvictionPolicy`](/docs/reference/wiki/configuration/cacheevictionpolicy/) | Cache eviction policies for KV cache management. |
| [`DifficultyEstimatorType`](/docs/reference/wiki/configuration/difficultyestimatortype/) | Types of difficulty estimators for curriculum learning. |
| [`DraftModelType`](/docs/reference/wiki/configuration/draftmodeltype/) | Types of draft models for speculative decoding. |
| [`ExplorationDecayType`](/docs/reference/wiki/configuration/explorationdecaytype/) | Type of exploration decay schedule. |
| [`GpuDiagnosticLevel`](/docs/reference/wiki/configuration/gpudiagnosticlevel/) | Verbosity level for GPU backend diagnostic output. |
| [`InferenceQuantizationMode`](/docs/reference/wiki/configuration/inferencequantizationmode/) | Specifies the weight quantization mode for inference optimization. |
| [`KVCachePrecisionMode`](/docs/reference/wiki/configuration/kvcacheprecisionmode/) | Controls the numeric precision of KV-cache storage. |
| [`KVCacheQuantizationMode`](/docs/reference/wiki/configuration/kvcachequantizationmode/) | Controls optional KV-cache quantization for inference. |
| [`SpeculationPolicy`](/docs/reference/wiki/configuration/speculationpolicy/) | Policies for enabling/disabling speculative decoding at runtime. |
| [`SpeculativeMethod`](/docs/reference/wiki/configuration/speculativemethod/) | Selects the speculative decoding method. |
| [`TrainingDiagnosticLevel`](/docs/reference/wiki/configuration/trainingdiagnosticlevel/) | Verbosity level for training-pipeline diagnostic output (gradient norms, optimizer step traces, tape replay events, etc.). |

## Delegates (2)

| Type | Summary |
|:-----|:--------|
| [`GpuDiagnosticSink`](/docs/reference/wiki/configuration/gpudiagnosticsink/) | Delegate that receives GPU backend diagnostic messages in lieu of `WriteLine`. |
| [`TrainingDiagnosticSink`](/docs/reference/wiki/configuration/trainingdiagnosticsink/) | Delegate that receives training-pipeline diagnostic events in lieu of the default `String)` path. |

## Options & Configuration (40)

| Type | Summary |
|:-----|:--------|
| [`AutoMLBudgetOptions`](/docs/reference/wiki/configuration/automlbudgetoptions/) | Configuration options that control AutoML compute budgets. |
| [`AutoMLEnsembleOptions`](/docs/reference/wiki/configuration/automlensembleoptions/) | Configuration options for AutoML ensembling. |
| [`AutoMLMultiFidelityOptions`](/docs/reference/wiki/configuration/automlmultifidelityoptions/) | Configuration options for multi-fidelity/ASHA AutoML search. |
| [`AutoMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/configuration/automloptions/) | Configuration options for running AutoML through the AiDotNet facade. |
| [`BenchmarkingOptions`](/docs/reference/wiki/configuration/benchmarkingoptions/) | Configuration options for running benchmarks through the AiDotNet facade. |
| [`CifarFederatedBenchmarkOptions`](/docs/reference/wiki/configuration/cifarfederatedbenchmarkoptions/) | Configuration options for running CIFAR-based federated benchmark suites. |
| [`CompetenceBasedOptions`](/docs/reference/wiki/configuration/competencebasedoptions/) | Options specific to competence-based curriculum learning. |
| [`CurriculumEarlyStoppingOptions`](/docs/reference/wiki/configuration/curriculumearlystoppingoptions/) | Early stopping options for curriculum learning. |
| [`CurriculumLearningOptions<T, TInput, TOutput>`](/docs/reference/wiki/configuration/curriculumlearningoptions/) | Configuration options for Curriculum Learning through the AiDotNet facade. |
| [`DenseNetConfiguration`](/docs/reference/wiki/configuration/densenetconfiguration/) | Configuration options for DenseNet neural network architectures. |
| [`EfficientNetConfiguration`](/docs/reference/wiki/configuration/efficientnetconfiguration/) | Configuration options for EfficientNet neural network architectures. |
| [`ExplorationScheduleConfig<T>`](/docs/reference/wiki/configuration/explorationscheduleconfig/) | Configuration for exploration schedule (epsilon decay for epsilon-greedy). |
| [`FederatedTabularBenchmarkOptions`](/docs/reference/wiki/configuration/federatedtabularbenchmarkoptions/) | Configuration options for federated tabular benchmark suites. |
| [`FederatedTextBenchmarkOptions`](/docs/reference/wiki/configuration/federatedtextbenchmarkoptions/) | Configuration options for federated text benchmark suites. |
| [`FederatedVisionBenchmarkOptions`](/docs/reference/wiki/configuration/federatedvisionbenchmarkoptions/) | Configuration options for federated vision benchmark suites. |
| [`GpuDiagnosticsConfig`](/docs/reference/wiki/configuration/gpudiagnosticsconfig/) | Process-global control for GPU backend diagnostic output visibility. |
| [`GpuDiagnosticsOptions`](/docs/reference/wiki/configuration/gpudiagnosticsoptions/) | Options for controlling GPU backend diagnostic output visibility. |
| [`InferenceOptimizationConfig`](/docs/reference/wiki/configuration/inferenceoptimizationconfig/) | Configuration for inference-time optimizations to maximize prediction throughput and efficiency. |
| [`JitCompilationConfig`](/docs/reference/wiki/configuration/jitcompilationconfig/) | Configuration for JIT (Just-In-Time) compilation of model forward/backward passes. |
| [`LeafFederatedBenchmarkOptions`](/docs/reference/wiki/configuration/leaffederatedbenchmarkoptions/) | Configuration options for running LEAF-backed federated benchmark suites. |
| [`MobileNetV2Configuration`](/docs/reference/wiki/configuration/mobilenetv2configuration/) | Configuration options for MobileNetV2 neural network architectures. |
| [`MobileNetV3Configuration`](/docs/reference/wiki/configuration/mobilenetv3configuration/) | Configuration options for MobileNetV3 neural network architectures. |
| [`PrioritizedReplayConfig<T>`](/docs/reference/wiki/configuration/prioritizedreplayconfig/) | Configuration for prioritized experience replay (PER). |
| [`RLAutoMLOptions<T>`](/docs/reference/wiki/configuration/rlautomloptions/) | Configuration options for running AutoML over reinforcement learning agents and hyperparameters. |
| [`RLCheckpointConfig`](/docs/reference/wiki/configuration/rlcheckpointconfig/) | Configuration for checkpointing during RL training. |
| [`RLEarlyStoppingConfig<T>`](/docs/reference/wiki/configuration/rlearlystoppingconfig/) | Configuration for early stopping during RL training. |
| [`RLEvaluationConfig`](/docs/reference/wiki/configuration/rlevaluationconfig/) | Configuration for evaluation during training. |
| [`RLTrainingOptions<T>`](/docs/reference/wiki/configuration/rltrainingoptions/) | Configuration options for reinforcement learning training loops via AiModelBuilder. |
| [`RedditFederatedBenchmarkOptions`](/docs/reference/wiki/configuration/redditfederatedbenchmarkoptions/) | Configuration options for running the Reddit federated benchmark suite. |
| [`ResNetConfiguration`](/docs/reference/wiki/configuration/resnetconfiguration/) | Configuration options for ResNet (Residual Network) neural network architectures. |
| [`RewardClippingConfig<T>`](/docs/reference/wiki/configuration/rewardclippingconfig/) | Configuration for reward clipping. |
| [`SelfPacedOptions`](/docs/reference/wiki/configuration/selfpacedoptions/) | Options specific to self-paced curriculum learning. |
| [`Sent140FederatedBenchmarkOptions`](/docs/reference/wiki/configuration/sent140federatedbenchmarkoptions/) | Configuration options for running the Sent140 LEAF federated benchmark suite. |
| [`ShakespeareFederatedBenchmarkOptions`](/docs/reference/wiki/configuration/shakespearefederatedbenchmarkoptions/) | Configuration options for running the Shakespeare LEAF federated benchmark suite. |
| [`StackOverflowFederatedBenchmarkOptions`](/docs/reference/wiki/configuration/stackoverflowfederatedbenchmarkoptions/) | Configuration options for running the StackOverflow federated benchmark suite. |
| [`SyntheticTabularFederatedBenchmarkOptions`](/docs/reference/wiki/configuration/synthetictabularfederatedbenchmarkoptions/) | Configuration options for the synthetic federated tabular benchmark suite. |
| [`TargetNetworkConfig<T>`](/docs/reference/wiki/configuration/targetnetworkconfig/) | Configuration for target network updates in DQN-family algorithms. |
| [`TrainingDiagnosticsConfig`](/docs/reference/wiki/configuration/trainingdiagnosticsconfig/) | Process-global control for training-pipeline diagnostic output (gradient norms, optimizer step traces, tape events). |
| [`VGGConfiguration`](/docs/reference/wiki/configuration/vggconfiguration/) | Configuration options for VGG neural network architectures. |
| [`YamlModelConfig`](/docs/reference/wiki/configuration/yamlmodelconfig/) | Root POCO that YAML/JSON config files deserialize into. |

## Helpers & Utilities (4)

| Type | Summary |
|:-----|:--------|
| [`GpuDiagnosticsLoggerExtensions`](/docs/reference/wiki/configuration/gpudiagnosticsloggerextensions/) | Extension methods that adapt `ILogger` into a `GpuDiagnosticSink`. |
| [`YamlConfigLoader`](/docs/reference/wiki/configuration/yamlconfigloader/) | Loads and deserializes YAML configuration files into strongly-typed configuration objects. |
| [`YamlDocsGenerator`](/docs/reference/wiki/configuration/yamldocsgenerator/) | Generates markdown reference documentation for the AiDotNet YAML configuration system. |
| [`YamlJsonSchema`](/docs/reference/wiki/configuration/yamljsonschema/) | Generates a JSON Schema for the AiDotNet YAML configuration system. |

## Attributes (1)

| Type | Summary |
|:-----|:--------|
| [`YamlConfigurableAttribute`](/docs/reference/wiki/configuration/yamlconfigurableattribute/) | Marks an interface or abstract base class as discoverable by the YAML configuration system. |

