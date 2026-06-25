---
title: "MetaLearnerBase<T, TInput, TOutput>"
description: "Unified base class for all meta-learning algorithms, providing both training infrastructure and shared algorithm utilities."
section: "API Reference"
---

`Base Classes` · `AiDotNet.MetaLearning`

Unified base class for all meta-learning algorithms, providing both training infrastructure
and shared algorithm utilities.

## How It Works

This base class follows the TimeSeriesModelBase pattern where the base class does heavy lifting
and concrete algorithm implementations override with their custom logic. It provides:

**For Algorithm Implementers:**
To create a new meta-learning algorithm:

1. Extend this base class
2. Set AlgorithmType in constructor
3. Override MetaTrainCore() with your algorithm's meta-update logic
4. Override AdaptCore() with your task adaptation strategy
5. All shared functionality (metrics, saving, evaluation) is handled automatically

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaLearnerBase(IFullModel<,,>,ILossFunction<>,IMetaLearnerOptions<>,IEpisodicDataLoader<,,>,IGradientBasedOptimizer<,,>,IGradientBasedOptimizer<,,>)` | Initializes a new instance of the MetaLearnerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `AlgorithmType` |  |
| `BaseModel` |  |
| `CurrentIteration` |  |
| `DefaultLossFunction` |  |
| `InnerLearningRate` |  |
| `Options` |  |
| `OuterLearningRate` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `AdaptAndEvaluate(MetaLearningTask<,,>)` |  |
| `ApplyGradients(Vector<>,Vector<>,Double)` | Applies gradients to update model parameters using simple SGD. |
| `ApplyOuterUpdate(Vector<>,List<Vector<>>,Double)` | Applies the standard outer-loop meta-update: averages meta-gradients and updates parameters. |
| `ApplyPerLayerGradients(IFullModel<,,>,Vector<>,Dictionary<Int32,Double>,Double)` | Applies gradients with per-layer learning rates using `ILayeredModel` metadata. |
| `AverageVectors(List<Vector<>>)` | Computes the element-wise average of a list of vectors. |
| `ClipGradients(Vector<>,Nullable<Double>)` | Clips gradients to prevent exploding gradients. |
| `CloneModel` | Clones the meta-model for task-specific adaptation. |
| `CompressVector(Vector<>,Int32)` | Compresses a Vector<T> into a smaller Vector<T> of target dimension using bucket averaging with tanh. |
| `ComputeAccuracy(,)` | Computes accuracy for classification tasks. |
| `ComputeGradients(IFullModel<,,>,,)` | Computes gradients of the loss with respect to model parameters using the model's built-in gradient computation via `IGradientComputable`. |
| `ComputeGradientsFallback(IFullModel<,,>,,)` | Fallback gradient computation for models that don't implement `IGradientComputable`. |
| `ComputeLossFromOutput(,)` | Computes loss from TOutput by converting to Vector<T> if needed. |
| `ComputeMean(List<>)` | Computes mean of a list of values. |
| `ComputeSecondOrderGradients(IFullModel<,,>,List<ValueTuple<,>>,,,)` | Computes second-order gradients for full MAML by backpropagating through the adaptation process. |
| `ConcatVectors(Vector<>,Vector<>)` | Concatenates two nullable vectors into a single vector. |
| `ConvertToVector()` | Converts TOutput to Vector<T> if possible. |
| `CosineSimilarity(Vector<>,Vector<>,Double)` | Computes cosine similarity between two Vector<T> instances. |
| `CreateTaskBatch(IReadOnlyList<MetaLearningTask<,,>>)` | Creates a TaskBatch from a list of MetaLearningTasks. |
| `DeepCopy` |  |
| `Evaluate(Int32)` |  |
| `Evaluate(TaskBatch<,,>)` |  |
| `FindArgmax(Vector<>)` | Finds the index of the maximum value in a vector (argmax). |
| `GetMetaModel` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `L2NormSquared(Vector<>)` | Computes the squared L2 norm of a Vector<T>. |
| `L2Normalize(Vector<>,Double)` | Returns a new L2-normalized copy of a Vector<T>. |
| `Load(String)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |
| `MetaTrainStep(ITaskSampler<,,>,Int32)` | Performs a single meta-training step using a task sampler. |
| `MetaTrainStep(Int32)` |  |
| `Predict()` | Predicts output by delegating to the meta-trained base model. |
| `Reset` |  |
| `SampleBeta(Double,Double)` | Samples from a Beta distribution using the Gamma distribution trick: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b)). |
| `SampleGamma(Double,Double)` | Samples from a Gamma distribution using the Marsaglia-Tsang method. |
| `SampleNormal(Double,Double)` | Samples from a standard normal distribution using the Box-Muller transform. |
| `Save(String)` |  |
| `ScalarSigmoid()` | Computes scalar sigmoid(x) = 1 / (1 + e^-x) using NumOps primitives. |
| `ScalarTanh()` | Computes scalar tanh(x) = (e^x - e^-x) / (e^x + e^-x) using NumOps primitives. |
| `ScaleVector(Vector<>,Double)` | Scales every element of a parameter vector by a scalar factor. |
| `SetMetaModel(IFullModel<,,>)` |  |
| `SetParameters(Vector<>)` |  |
| `Softmax(Vector<>)` | Computes softmax over a Vector<T> of logits (SIMD-accelerated via Engine). |
| `StandardAuxLoss(TaskBatch<,,>)` | Computes the standard auxiliary loss for SPSA updates: average query loss across all tasks. |
| `StandardInnerAdapt(Vector<>,IMetaLearningTask<,,>,Int32,Double)` | Performs the standard MAML-style inner loop: K gradient steps on support data starting from initParams. |
| `ToMetaLearningTask(MetaLearningTask<,,>)` | Converts a MetaLearningTask to IMetaLearningTask. |
| `Train` |  |
| `Train(,)` | Trains the meta-learner on a single input-output pair. |
| `UpdateAuxiliaryParamsSPSA(TaskBatch<,,>,Vector<>,Double,Func<TaskBatch<,,>,Double>,Int32,Double)` | Updates auxiliary parameters using multi-sample SPSA (Simultaneous Perturbation Stochastic Approximation) with a caller-provided loss function. |
| `VectorExp(Vector<>)` | Computes element-wise exponential (SIMD-accelerated via Engine). |
| `VectorLog(Vector<>)` | Computes element-wise natural logarithm (SIMD-accelerated via Engine). |
| `VectorTanh(Vector<>)` | Computes element-wise tanh (SIMD-accelerated via Engine). |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DataLoader` | Episodic data loader for sampling meta-learning tasks. |
| `InnerOptimizer` | Optimizer for task adaptation (inner loop). |
| `LossFunction` | The loss function used to evaluate task performance. |
| `MetaModel` | The model being meta-trained. |
| `MetaOptimizer` | Optimizer for meta-parameter updates (outer loop). |
| `RandomGenerator` | Random number generator for stochastic operations. |
| `_currentIteration` | Current meta-training iteration count. |
| `_gradientFallbackWarningEmitted` | Tracks whether the expensive gradient fallback warning has been emitted. |
| `_options` | Configuration options for meta-learning. |

