---
title: "Curriculum Learning"
description: "All 45 public types in the AiDotNet.curriculumlearning namespace, organized by kind."
section: "API Reference"
---

**45** public types in this namespace, organized by kind.

## Models & Types (22)

| Type | Summary |
|:-----|:--------|
| [`CompetenceBasedScheduler<T>`](/docs/reference/wiki/curriculumlearning/competencebasedscheduler/) | Curriculum scheduler that advances based on model competence/mastery. |
| [`ComplexityBasedDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/complexitybaseddifficultyestimator/) | Difficulty estimator based on sample complexity metrics. |
| [`ConfidenceBasedDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/confidencebaseddifficultyestimator/) | Difficulty estimator based on model prediction confidence. |
| [`CosineScheduler<T>`](/docs/reference/wiki/curriculumlearning/cosinescheduler/) | Curriculum scheduler with cosine annealing curve. |
| [`CurriculumEpochMetrics<T>`](/docs/reference/wiki/curriculumlearning/curriculumepochmetrics/) | Metrics from a curriculum learning epoch. |
| [`CurriculumLearner<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/curriculumlearner/) | Main orchestrator for curriculum learning that coordinates difficulty estimation, scheduling, and model training. |
| [`CurriculumLearningResult<T>`](/docs/reference/wiki/curriculumlearning/curriculumlearningresult/) | Result of curriculum learning training. |
| [`CurriculumPhaseCompletedEventArgs<T>`](/docs/reference/wiki/curriculumlearning/curriculumphasecompletedeventargs/) | Event arguments for curriculum phase completion events. |
| [`CurriculumPhaseEventArgs<T>`](/docs/reference/wiki/curriculumlearning/curriculumphaseeventargs/) | Event arguments for curriculum phase events. |
| [`CurriculumPhaseResult<T>`](/docs/reference/wiki/curriculumlearning/curriculumphaseresult/) | Result of a single curriculum phase. |
| [`CurriculumProgressionEntry<T>`](/docs/reference/wiki/curriculumlearning/curriculumprogressionentry/) | Entry in the curriculum progression history. |
| [`CurriculumTrainingCompletedEventArgs<T>`](/docs/reference/wiki/curriculumlearning/curriculumtrainingcompletedeventargs/) | Event arguments for training completion. |
| [`EnsembleDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/ensembledifficultyestimator/) | Difficulty estimator that combines multiple estimators for robust difficulty estimation. |
| [`ExpertDefinedDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/expertdefineddifficultyestimator/) | Difficulty estimator using pre-defined or expert-provided difficulty scores. |
| [`ExponentialScheduler<T>`](/docs/reference/wiki/curriculumlearning/exponentialscheduler/) | Curriculum scheduler with exponential (slow start) progression. |
| [`LinearScheduler<T>`](/docs/reference/wiki/curriculumlearning/linearscheduler/) | Curriculum scheduler with linear progression from easy to hard samples. |
| [`LossBasedDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/lossbaseddifficultyestimator/) | Difficulty estimator based on training loss. |
| [`PolynomialScheduler<T>`](/docs/reference/wiki/curriculumlearning/polynomialscheduler/) | Curriculum scheduler with polynomial progression curve. |
| [`SelfPacedScheduler<T>`](/docs/reference/wiki/curriculumlearning/selfpacedscheduler/) | Self-paced curriculum scheduler that adapts sample selection based on model performance. |
| [`SmoothedLossDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/smoothedlossdifficultyestimator/) | Loss-based difficulty estimator with moving average smoothing. |
| [`StepScheduler<T>`](/docs/reference/wiki/curriculumlearning/stepscheduler/) | Curriculum scheduler with discrete step-based progression. |
| [`TransferBasedDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/transferbaseddifficultyestimator/) | Difficulty estimator based on transfer learning from a simpler "teacher" model. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`CurriculumSchedulerBase<T>`](/docs/reference/wiki/curriculumlearning/curriculumschedulerbase/) | Abstract base class for curriculum schedulers providing common functionality. |
| [`DifficultyEstimatorBase<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/difficultyestimatorbase/) | Base class for difficulty estimators. |

## Interfaces (10)

| Type | Summary |
|:-----|:--------|
| [`ICompetenceBasedScheduler<T>`](/docs/reference/wiki/curriculumlearning/icompetencebasedscheduler/) | Interface for competence-based curriculum schedulers. |
| [`IConfidentDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/iconfidentdifficultyestimator/) | Interface for difficulty estimators that can provide confidence in their estimates. |
| [`ICurriculumLearnerConfigBuilder<T>`](/docs/reference/wiki/curriculumlearning/icurriculumlearnerconfigbuilder/) | Builder for curriculum learner configuration. |
| [`ICurriculumLearnerConfig<T>`](/docs/reference/wiki/curriculumlearning/icurriculumlearnerconfig/) | Configuration interface for curriculum learning. |
| [`ICurriculumLearner<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/icurriculumlearner/) | Interface for curriculum learning trainers that train models using a structured curriculum. |
| [`ICurriculumScheduler<T>`](/docs/reference/wiki/curriculumlearning/icurriculumscheduler/) | Interface for curriculum schedulers that control training progression. |
| [`IDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/idifficultyestimator/) | Interface for estimating the difficulty of training samples. |
| [`IEnsembleDifficultyEstimator<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/iensembledifficultyestimator/) | Interface for ensemble difficulty estimators that combine multiple estimators. |
| [`IProbabilisticModel<T, TInput, TOutput>`](/docs/reference/wiki/curriculumlearning/iprobabilisticmodel/) | Interface for models that provide probability predictions. |
| [`ISelfPacedScheduler<T>`](/docs/reference/wiki/curriculumlearning/iselfpacedscheduler/) | Interface for self-paced curriculum schedulers that adapt based on model performance. |

## Enums (9)

| Type | Summary |
|:-----|:--------|
| [`CompetenceMetricType`](/docs/reference/wiki/curriculumlearning/competencemetrictype/) | Type of competence metric for curriculum advancement. |
| [`ComplexityMetric`](/docs/reference/wiki/curriculumlearning/complexitymetric/) | Complexity metric types. |
| [`ConfidenceMetricType`](/docs/reference/wiki/curriculumlearning/confidencemetrictype/) | Type of confidence metric used for difficulty estimation. |
| [`CurriculumScheduleType`](/docs/reference/wiki/curriculumlearning/curriculumscheduletype/) | Types of curriculum scheduling strategies. |
| [`CurriculumVerbosity`](/docs/reference/wiki/curriculumlearning/curriculumverbosity/) | Verbosity levels for curriculum learning. |
| [`DifficultyEstimationType`](/docs/reference/wiki/curriculumlearning/difficultyestimationtype/) | Types of difficulty estimation methods. |
| [`EnsembleCombinationMethod`](/docs/reference/wiki/curriculumlearning/ensemblecombinationmethod/) | Methods for combining multiple difficulty estimates. |
| [`SelfPaceRegularizer`](/docs/reference/wiki/curriculumlearning/selfpaceregularizer/) | Type of self-pace regularizer for sample weighting. |
| [`TransferDifficultyMode`](/docs/reference/wiki/curriculumlearning/transferdifficultymode/) | Mode for transfer-based difficulty calculation. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`CurriculumLearnerConfig<T>`](/docs/reference/wiki/curriculumlearning/curriculumlearnerconfig/) | Configuration for curriculum learning training. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`CurriculumLearnerConfigBuilder<T>`](/docs/reference/wiki/curriculumlearning/curriculumlearnerconfigbuilder/) | Fluent builder for curriculum learning configuration. |

