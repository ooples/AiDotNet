---
title: "Continual Learning"
description: "All 66 public types in the AiDotNet.continuallearning namespace, organized by kind."
section: "API Reference"
---

**66** public types in this namespace, organized by kind.

## Models & Types (35)

| Type | Summary |
|:-----|:--------|
| [`AveragedGEM<T>`](/docs/reference/wiki/continuallearning/averagedgem/) | Implements Averaged Gradient Episodic Memory (A-GEM) for continual learning. |
| [`BufferStatistics`](/docs/reference/wiki/continuallearning/bufferstatistics/) | Statistics about the experience replay buffer. |
| [`ContinualEvaluationResult<T>`](/docs/reference/wiki/continuallearning/continualevaluationresult/) | Comprehensive evaluation result across all learned tasks. |
| [`ContinualLearningResult<T>`](/docs/reference/wiki/continuallearning/continuallearningresult/) | Result from training on a single task in continual learning. |
| [`DataPoint<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/datapoint/) | Represents a single data point with input and output. |
| [`EWCTrainer<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/ewctrainer/) | Continual learning trainer using Elastic Weight Consolidation (EWC). |
| [`ElasticWeightConsolidation<T>`](/docs/reference/wiki/continuallearning/elasticweightconsolidation/) | Implements Elastic Weight Consolidation (EWC) for continual learning. |
| [`ElasticWeightConsolidation<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/elasticweightconsolidation-2/) |  |
| [`EpochEventArgs<T>`](/docs/reference/wiki/continuallearning/epocheventargs/) | Event arguments for epoch completion events. |
| [`ExpectedGradientLength<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/expectedgradientlength/) | Expected Gradient Length (EGL) strategy for continual learning. |
| [`ExperienceDataPoint<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/experiencedatapoint/) | Alias for DataPoint used in experience replay contexts. |
| [`ExperienceReplayBuffer<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/experiencereplaybuffer/) | A memory buffer for storing examples from previous tasks for experience replay. |
| [`ExperienceReplay<T>`](/docs/reference/wiki/continuallearning/experiencereplay/) | Implements Experience Replay for continual learning. |
| [`GEMTrainer<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/gemtrainer/) | Continual learning trainer using Gradient Episodic Memory (GEM). |
| [`GenerativeReplay<T>`](/docs/reference/wiki/continuallearning/generativereplay/) | Implements Generative Replay (also known as Deep Generative Replay) for continual learning. |
| [`GradientEpisodicMemory<T>`](/docs/reference/wiki/continuallearning/gradientepisodicmemory/) | Implements Gradient Episodic Memory (GEM) for continual learning. |
| [`GradientEpisodicMemory<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/gradientepisodicmemory-2/) | Gradient Episodic Memory (GEM) strategy for continual learning with gradient constraints. |
| [`LearningWithoutForgetting<T>`](/docs/reference/wiki/continuallearning/learningwithoutforgetting/) | Implements Learning without Forgetting (LwF) for continual learning. |
| [`LearningWithoutForgetting<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/learningwithoutforgetting-2/) | Learning without Forgetting (LwF) strategy for continual learning. |
| [`LwFTrainer<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/lwftrainer/) | Continual learning trainer using Learning without Forgetting (LwF). |
| [`MASTrainer<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/mastrainer/) | Continual learning trainer using Memory Aware Synapses (MAS). |
| [`MemoryAwareSynapses<T>`](/docs/reference/wiki/continuallearning/memoryawaresynapses/) | Implements Memory Aware Synapses (MAS) for continual learning. |
| [`MemoryAwareSynapses<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/memoryawaresynapses-2/) | Memory Aware Synapses (MAS) strategy for continual learning. |
| [`OnlineEWC<T>`](/docs/reference/wiki/continuallearning/onlineewc/) | Implements Online Elastic Weight Consolidation (Online EWC) for continual learning. |
| [`PackNet<T>`](/docs/reference/wiki/continuallearning/packnet/) | Implements PackNet for continual learning through parameter isolation. |
| [`PackNet<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/packnet-2/) | PackNet strategy for continual learning through network pruning. |
| [`ProgressiveNeuralNetworks<T>`](/docs/reference/wiki/continuallearning/progressiveneuralnetworks/) | Implements Progressive Neural Networks for continual learning. |
| [`ProgressiveNeuralNetworks<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/progressiveneuralnetworks-2/) | Progressive Neural Networks (PNN) strategy for continual learning. |
| [`SITrainer<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/sitrainer/) | Continual learning trainer using Synaptic Intelligence (SI). |
| [`SynapticIntelligence<T>`](/docs/reference/wiki/continuallearning/synapticintelligence-2/) | Implements Synaptic Intelligence (SI) for continual learning. |
| [`SynapticIntelligence<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/synapticintelligence/) | Synaptic Intelligence (SI) strategy for continual learning. |
| [`TaskCompletedEventArgs<T>`](/docs/reference/wiki/continuallearning/taskcompletedeventargs/) | Event arguments for task completion events. |
| [`TaskEvaluationResult<T>`](/docs/reference/wiki/continuallearning/taskevaluationresult/) | Result from evaluating model performance on a single task. |
| [`TaskEventArgs`](/docs/reference/wiki/continuallearning/taskeventargs/) | Event arguments for task events. |
| [`VariationalContinualLearning<T>`](/docs/reference/wiki/continuallearning/variationalcontinuallearning/) | Implements Variational Continual Learning (VCL) for Bayesian continual learning. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`ContinualLearnerBase<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/continuallearnerbase/) | Base class for continual learning trainers that provides common functionality. |
| [`ContinualLearningStrategyBase<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/continuallearningstrategybase/) | Abstract base class for continual learning strategies providing common functionality. |

## Interfaces (8)

| Type | Summary |
|:-----|:--------|
| [`IContinualLearnerConfig<T>`](/docs/reference/wiki/continuallearning/icontinuallearnerconfig/) | Configuration interface for continual learning algorithms. |
| [`IContinualLearner<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/icontinuallearner/) | Interface for continual learning trainers that can learn multiple tasks sequentially. |
| [`IContinualLearningStrategy<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/icontinuallearningstrategy/) | Strategy interface for continual learning algorithms. |
| [`IDistillationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/idistillationstrategy/) | Extended strategy interface for knowledge distillation-based strategies. |
| [`IGenerativeModel<T>`](/docs/reference/wiki/continuallearning/igenerativemodel/) | Interface for generative models used with GenerativeReplay. |
| [`IGradientCapable<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/igradientcapable/) | Interface for models that support gradient computation. |
| [`IGradientConstraintStrategy<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/igradientconstraintstrategy/) | Extended strategy interface for gradient-based constraint strategies. |
| [`IMemoryBasedStrategy<T, TInput, TOutput>`](/docs/reference/wiki/continuallearning/imemorybasedstrategy/) | Extended strategy interface for strategies that store task examples. |

## Enums (6)

| Type | Summary |
|:-----|:--------|
| [`BufferStrategy<T>`](/docs/reference/wiki/continuallearning/bufferstrategy/) | Defines the buffer management strategy. |
| [`DistillationLossType`](/docs/reference/wiki/continuallearning/distillationlosstype/) | Types of distillation loss functions. |
| [`ImportanceAccumulationMode`](/docs/reference/wiki/continuallearning/importanceaccumulationmode/) | Mode for accumulating importance across tasks. |
| [`MASImportanceMode`](/docs/reference/wiki/continuallearning/masimportancemode/) | Mode for computing parameter importance in MAS. |
| [`MemorySamplingStrategy`](/docs/reference/wiki/continuallearning/memorysamplingstrategy/) | Memory sampling strategies for experience replay. |
| [`ReplaySamplingStrategy`](/docs/reference/wiki/continuallearning/replaysamplingstrategy/) | Strategy for sampling during replay (training time). |

## Options & Configuration (14)

| Type | Summary |
|:-----|:--------|
| [`ContinualLearnerConfig<T>`](/docs/reference/wiki/continuallearning/continuallearnerconfig/) | Production-ready configuration for continual learning algorithms. |
| [`EGLOptions<T>`](/docs/reference/wiki/continuallearning/egloptions/) | Configuration options for Expected Gradient Length strategy. |
| [`EWCOptions<T>`](/docs/reference/wiki/continuallearning/ewcoptions/) | Configuration options for Elastic Weight Consolidation. |
| [`EWCTrainerOptions<T>`](/docs/reference/wiki/continuallearning/ewctraineroptions/) | Configuration options for the EWC trainer. |
| [`GEMOptions<T>`](/docs/reference/wiki/continuallearning/gemoptions/) | Configuration options for Gradient Episodic Memory strategies. |
| [`GEMTrainerOptions<T>`](/docs/reference/wiki/continuallearning/gemtraineroptions/) | Configuration options for the GEM trainer. |
| [`LwFOptions<T>`](/docs/reference/wiki/continuallearning/lwfoptions/) | Configuration options for Learning without Forgetting. |
| [`LwFTrainerOptions<T>`](/docs/reference/wiki/continuallearning/lwftraineroptions/) | Configuration options for the LwF trainer. |
| [`MASOptions<T>`](/docs/reference/wiki/continuallearning/masoptions/) | Configuration options for Memory Aware Synapses. |
| [`MASTrainerOptions<T>`](/docs/reference/wiki/continuallearning/mastraineroptions/) | Configuration options for the MAS trainer. |
| [`PNNOptions<T>`](/docs/reference/wiki/continuallearning/pnnoptions/) | Configuration options for Progressive Neural Networks strategy. |
| [`PackNetOptions<T>`](/docs/reference/wiki/continuallearning/packnetoptions/) | Configuration options for PackNet strategy. |
| [`SIOptions<T>`](/docs/reference/wiki/continuallearning/sioptions/) | Configuration options for Synaptic Intelligence. |
| [`SITrainerOptions<T>`](/docs/reference/wiki/continuallearning/sitraineroptions/) | Configuration options for the SI trainer. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`DatasetExtensions`](/docs/reference/wiki/continuallearning/datasetextensions/) | Extension methods for IDataset to convert to DataPoints. |

