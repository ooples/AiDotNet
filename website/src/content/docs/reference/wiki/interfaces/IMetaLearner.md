---
title: "IMetaLearner<T, TInput, TOutput>"
description: "Unified interface for meta-learning algorithms that train models to quickly adapt to new tasks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Unified interface for meta-learning algorithms that train models to quickly adapt to new tasks.

## For Beginners

Meta-learning is like teaching someone how to learn, not just what to learn.

Traditional vs Meta-Learning:

- **Traditional:** Train on thousands of cat/dog images → classify cats vs dogs well
- **Meta-Learning:** Train on many classification tasks → learn ANY new category from 5 examples

Real-world applications:

- Few-shot image classification (recognize new objects from 1-5 images)
- Rapid robot adaptation (new environments with minimal data)
- Personalized recommendations (adapt to new users quickly)
- Drug discovery (predict properties of new molecules)

## How It Works

This is the unified interface for all meta-learning algorithms in the framework.
It combines both training infrastructure and algorithm capabilities, enabling
seamless integration with AiModelBuilder while supporting all 17 meta-learning
algorithms (MAML, Reptile, ProtoNets, LEO, MetaOptNet, etc.).

**Architecture - Two-Loop Optimization:****Inner Loop (Task Adaptation):**

- Given: New task with support set (K examples per class)
- Process: Few gradient steps (1-10) to adapt model
- Output: Task-specific adapted parameters
- Goal: Quickly learn this specific task

**Outer Loop (Meta-Optimization):**

- Given: Batch of tasks from task distribution
- Process: For each task, adapt (inner loop) and evaluate on query set
- Output: Updated meta-parameters
- Goal: Learn parameters that enable fast adaptation across all tasks

This two-loop structure is what enables "learning to learn."

**Production Considerations:**

- Use MetaTrainStep() for training loop with proper batch sizes (2-32 tasks)
- Monitor Evaluate() metrics every N iterations to detect overfitting
- Use AdaptAndEvaluate() for deployment to quickly adapt to new tasks
- Save/Load models after meta-training for deployment
- Thread Safety: Not thread-safe, use separate instances for concurrent training

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets the number of adaptation steps to perform during task adaptation (inner loop). |
| `AlgorithmType` | Gets the type of meta-learning algorithm. |
| `BaseModel` | Gets the base model being meta-trained. |
| `CurrentIteration` | Gets the current meta-training iteration count. |
| `InnerLearningRate` | Gets the learning rate used for task adaptation (inner loop). |
| `Options` | Gets the meta-learner options (configuration). |
| `OuterLearningRate` | Gets the learning rate used for meta-learning (outer loop). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the model to a new task using its support set. |
| `AdaptAndEvaluate(MetaLearningTask<,,>)` | Adapts the model to a specific task and evaluates adaptation quality. |
| `Evaluate(Int32)` | Evaluates meta-learning performance on multiple held-out tasks. |
| `Evaluate(TaskBatch<,,>)` | Evaluates the meta-learning algorithm on a batch of tasks. |
| `GetMetaModel` | Gets the current meta-model. |
| `Load(String)` | Loads a previously meta-trained model from disk. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step on a batch of tasks. |
| `MetaTrainStep(Int32)` | Performs one meta-training step (outer loop update) using the episodic data loader. |
| `Reset` | Resets the meta-learner to initial untrained state. |
| `Save(String)` | Saves the meta-trained model to disk for later deployment. |
| `SetMetaModel(IFullModel<,,>)` | Sets the base model for this meta-learning algorithm. |
| `Train` | Trains the meta-learner using the configuration specified during construction. |

