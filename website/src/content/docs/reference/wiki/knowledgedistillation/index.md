---
title: "Knowledge Distillation"
description: "All 56 public types in the AiDotNet.knowledgedistillation namespace, organized by kind."
section: "API Reference"
---

**56** public types in this namespace, organized by kind.

## Models & Types (34)

| Type | Summary |
|:-----|:--------|
| [`AccuracyBasedAdaptiveStrategy<T>`](/docs/reference/wiki/knowledgedistillation/accuracybasedadaptivestrategy/) | Adaptive distillation strategy that adjusts temperature based on student accuracy. |
| [`AdaptiveTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/adaptiveteachermodel/) | Adaptive teacher model that wraps a base teacher and provides its logits. |
| [`AttentionDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/attentiondistillationstrategy/) | Implements attention-based knowledge distillation for transformer models. |
| [`CheckpointMetadata`](/docs/reference/wiki/knowledgedistillation/checkpointmetadata/) | Metadata about a saved checkpoint. |
| [`ConfidenceBasedAdaptiveStrategy<T>`](/docs/reference/wiki/knowledgedistillation/confidencebasedadaptivestrategy/) | Adaptive distillation strategy that adjusts temperature based on student confidence. |
| [`ContrastiveDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/contrastivedistillationstrategy/) | Implements Contrastive Representation Distillation (CRD) which transfers knowledge through contrastive learning of sample relationships rather than just matching outputs. |
| [`CurriculumTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/curriculumteachermodel/) | Curriculum teacher that wraps a base teacher for curriculum learning scenarios. |
| [`DistillationCheckpointManager<T>`](/docs/reference/wiki/knowledgedistillation/distillationcheckpointmanager/) | Manages checkpointing during knowledge distillation training. |
| [`DistillationForwardResult<T>`](/docs/reference/wiki/knowledgedistillation/distillationforwardresult/) | Encapsulates the result of a forward pass during knowledge distillation training. |
| [`DistillationLoss<T>`](/docs/reference/wiki/knowledgedistillation/distillationloss/) |  |
| [`DistributedTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/distributedteachermodel/) | Distributed teacher model that aggregates predictions from multiple distributed workers. |
| [`EasyToHardCurriculumStrategy<T>`](/docs/reference/wiki/knowledgedistillation/easytohardcurriculumstrategy/) | Curriculum distillation strategy that progresses from easy to hard samples. |
| [`EnsembleTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/ensembleteachermodel/) | Ensemble teacher model that combines predictions from multiple teacher models. |
| [`EntropyBasedAdaptiveStrategy<T>`](/docs/reference/wiki/knowledgedistillation/entropybasedadaptivestrategy/) | Adaptive distillation strategy that adjusts temperature based on prediction entropy. |
| [`FactorTransferDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/factortransferdistillationstrategy/) |  |
| [`FeatureDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/featuredistillationstrategy/) | Implements feature-based knowledge distillation (FitNets) where the student learns to match the teacher's intermediate layer representations. |
| [`FlowBasedDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/flowbaseddistillationstrategy/) | Flow-based distillation that matches the information flow between layers. |
| [`HardToEasyCurriculumStrategy<T>`](/docs/reference/wiki/knowledgedistillation/hardtoeasycurriculumstrategy/) | Curriculum distillation strategy that progresses from hard to easy samples. |
| [`HybridDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/hybriddistillationstrategy/) | Hybrid distillation strategy that combines multiple distillation strategies with configurable weights. |
| [`IntermediateActivations<T>`](/docs/reference/wiki/knowledgedistillation/intermediateactivations/) | Stores intermediate layer activations collected during a forward pass. |
| [`KnowledgeDistillationTrainer<T>`](/docs/reference/wiki/knowledgedistillation/knowledgedistillationtrainer/) | Standard knowledge distillation trainer that uses a fixed teacher model to train a student. |
| [`MultiModalTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/multimodalteachermodel/) | Multi-modal teacher that combines multiple input modalities (vision, text, audio). |
| [`NeuronSelectivityDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/neuronselectivitydistillationstrategy/) | Neuron selectivity distillation that transfers the activation patterns and selectivity of individual neurons. |
| [`OnlineTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/onlineteachermodel/) | Online teacher model that updates its parameters during student training. |
| [`PretrainedTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/pretrainedteachermodel/) | Pretrained teacher model from external source (e.g., ImageNet, BERT). |
| [`ProbabilisticDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/probabilisticdistillationstrategy/) | Probabilistic distillation that transfers distributional knowledge by matching statistical properties. |
| [`QuantizedTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/quantizedteachermodel/) | Quantized teacher model with reduced precision for efficient deployment. |
| [`RelationalDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/relationaldistillationstrategy/) |  |
| [`SelfDistillationTrainer<T>`](/docs/reference/wiki/knowledgedistillation/selfdistillationtrainer/) | Implements self-distillation where a model acts as its own teacher to improve calibration and generalization. |
| [`SelfTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/selfteachermodel/) | Self teacher model that uses the student's own predictions from earlier training. |
| [`SimilarityPreservingStrategy<T>`](/docs/reference/wiki/knowledgedistillation/similaritypreservingstrategy/) | Similarity-preserving distillation that preserves pairwise similarity structure. |
| [`TeacherModelWrapper<T>`](/docs/reference/wiki/knowledgedistillation/teachermodelwrapper/) | Wraps an existing trained IFullModel to act as a teacher for knowledge distillation. |
| [`TransformerTeacherModel<T>`](/docs/reference/wiki/knowledgedistillation/transformerteachermodel/) | Transformer-based teacher model that provides logits from transformer architectures. |
| [`VariationalDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/variationaldistillationstrategy/) | Variational distillation based on variational inference principles and information theory. |

## Base Classes (5)

| Type | Summary |
|:-----|:--------|
| [`AdaptiveDistillationStrategyBase<T>`](/docs/reference/wiki/knowledgedistillation/adaptivedistillationstrategybase/) | Abstract base class for adaptive distillation strategies with performance tracking. |
| [`CurriculumDistillationStrategyBase<T>`](/docs/reference/wiki/knowledgedistillation/curriculumdistillationstrategybase/) | Abstract base class for curriculum distillation strategies with progressive difficulty adjustment. |
| [`DistillationStrategyBase<T>`](/docs/reference/wiki/knowledgedistillation/distillationstrategybase/) | Abstract base class for knowledge distillation strategies. |
| [`KnowledgeDistillationTrainerBase<T, TInput, TOutput>`](/docs/reference/wiki/knowledgedistillation/knowledgedistillationtrainerbase/) | Abstract base class for all knowledge distillation trainers. |
| [`TeacherModelBase<TInput, TOutput, T>`](/docs/reference/wiki/knowledgedistillation/teachermodelbase/) | Abstract base class for teacher models used in knowledge distillation. |

## Interfaces (2)

| Type | Summary |
|:-----|:--------|
| [`IAdaptiveDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/iadaptivedistillationstrategy/) | Interface for adaptive distillation strategies that adjust temperature based on student performance. |
| [`ICurriculumDistillationStrategy<T>`](/docs/reference/wiki/knowledgedistillation/icurriculumdistillationstrategy/) | Interface for curriculum distillation strategies that progressively adjust training difficulty. |

## Enums (11)

| Type | Summary |
|:-----|:--------|
| [`AggregationMode`](/docs/reference/wiki/knowledgedistillation/aggregationmode/) | Specifies how multiple teacher outputs are combined into a single supervision signal. |
| [`AttentionMatchingMode`](/docs/reference/wiki/knowledgedistillation/attentionmatchingmode/) | Defines how to match attention patterns between teacher and student. |
| [`ContrastiveMode`](/docs/reference/wiki/knowledgedistillation/contrastivemode/) | Defines the contrastive learning mode. |
| [`CurriculumStrategy`](/docs/reference/wiki/knowledgedistillation/curriculumstrategy/) | Defines the curriculum learning strategy direction. |
| [`EnsembleAggregationMode`](/docs/reference/wiki/knowledgedistillation/ensembleaggregationmode/) | Defines how ensemble predictions are aggregated. |
| [`FactorMode`](/docs/reference/wiki/knowledgedistillation/factormode/) |  |
| [`OnlineUpdateMode`](/docs/reference/wiki/knowledgedistillation/onlineupdatemode/) | Defines how an online teacher model is updated during training. |
| [`ProbabilisticMode`](/docs/reference/wiki/knowledgedistillation/probabilisticmode/) |  |
| [`RelationalDistanceMetric`](/docs/reference/wiki/knowledgedistillation/relationaldistancemetric/) | Distance metrics for relational knowledge distillation. |
| [`SelectivityMetric`](/docs/reference/wiki/knowledgedistillation/selectivitymetric/) |  |
| [`VariationalMode`](/docs/reference/wiki/knowledgedistillation/variationalmode/) |  |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`DistillationCheckpointConfig`](/docs/reference/wiki/knowledgedistillation/distillationcheckpointconfig/) | Configuration for distillation checkpoint management. |

## Helpers & Utilities (3)

| Type | Summary |
|:-----|:--------|
| [`DistillationStrategyFactory<T>`](/docs/reference/wiki/knowledgedistillation/distillationstrategyfactory/) | Factory for creating distillation strategies from enums and configurations. |
| [`StrategyBuilder<T>`](/docs/reference/wiki/knowledgedistillation/strategybuilder/) | Fluent builder for configuring distillation strategies with custom parameters. |
| [`TeacherModelFactory<T>`](/docs/reference/wiki/knowledgedistillation/teachermodelfactory/) | Factory for creating teacher models from enums and configurations. |

