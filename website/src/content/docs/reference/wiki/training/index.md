---
title: "Training"
description: "All 15 public types in the AiDotNet.training namespace, organized by kind."
section: "API Reference"
---

**15** public types in this namespace, organized by kind.

## Models & Types (4)

| Type | Summary |
|:-----|:--------|
| [`MemorySavingsEstimate`](/docs/reference/wiki/training/memorysavingsestimate/) | Memory savings estimate from optimization techniques. |
| [`Trainer<T>`](/docs/reference/wiki/training/trainer/) | Default trainer that delegates to the model's built-in `Train()` method each epoch. |
| [`TrainingMemoryManager<T>`](/docs/reference/wiki/training/trainingmemorymanager/) | Manages memory optimization during neural network training including gradient checkpointing, activation pooling, and model sharding. |
| [`TrainingResult<T>`](/docs/reference/wiki/training/trainingresult/) | Contains the results of a training run, including the trained model, loss history, and metadata. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`TrainerBase<T>`](/docs/reference/wiki/training/trainerbase/) | Abstract base class for all trainers, providing shared infrastructure for configuration-driven training pipelines. |

## Options & Configuration (7)

| Type | Summary |
|:-----|:--------|
| [`DatasetConfig`](/docs/reference/wiki/training/datasetconfig/) | Configuration for the dataset section of a training recipe. |
| [`LossFunctionConfig`](/docs/reference/wiki/training/lossfunctionconfig/) | Configuration for the loss function section of a training recipe. |
| [`ModelConfig`](/docs/reference/wiki/training/modelconfig/) | Configuration for the model section of a training recipe. |
| [`OptimizerConfig`](/docs/reference/wiki/training/optimizerconfig/) | Configuration for the optimizer section of a training recipe. |
| [`TrainerSettings`](/docs/reference/wiki/training/trainersettings/) | Configuration for the trainer behavior section of a training recipe. |
| [`TrainingMemoryConfig`](/docs/reference/wiki/training/trainingmemoryconfig/) | Configuration for training memory management including gradient checkpointing, activation pooling, and model sharding. |
| [`TrainingRecipeConfig`](/docs/reference/wiki/training/trainingrecipeconfig/) | Root configuration object for a complete training recipe defined in YAML. |

## Helpers & Utilities (3)

| Type | Summary |
|:-----|:--------|
| [`CompiledTapeTrainingStep<T>`](/docs/reference/wiki/training/compiledtapetrainingstep/) | Compiled training step — auto-compiles the forward + backward pass on the first step, then replays the compiled plan on subsequent steps for near-zero overhead training. |
| [`LossFunctionFactory<T>`](/docs/reference/wiki/training/lossfunctionfactory/) | Factory for creating loss function instances from `LossType` enum values. |
| [`TapeTrainingStep<T>`](/docs/reference/wiki/training/tapetrainingstep/) | Provides PyTorch-style training step using tape-based automatic differentiation, with two-level caching for parameter collection that outperforms PyTorch's `model.parameters()` which rebuilds the full list on every call. |

