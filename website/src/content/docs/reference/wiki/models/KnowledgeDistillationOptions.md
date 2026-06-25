---
title: "KnowledgeDistillationOptions<T, TInput, TOutput>"
description: "Configuration options for knowledge distillation training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for knowledge distillation training.

## For Beginners

This class configures how knowledge distillation works.
Think of it as the "settings" for transferring knowledge from a large teacher model
to a smaller student model.

## How It Works

**Quick Start Example:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the alpha parameter balancing hard and soft loss. |
| `AttentionLayers` | Gets or sets attention layer names (if using attention-based distillation). |
| `AttentionWeight` | Gets or sets weight for attention loss (if using attention-based distillation). |
| `BatchSize` | Gets or sets the batch size for training. |
| `CheckpointDirectory` | Gets or sets checkpoint directory path (if checkpoints are enabled). |
| `CheckpointFrequency` | Gets or sets checkpoint frequency (save every N epochs). |
| `EMADecay` | Gets or sets the EMA decay rate (if using EMA). |
| `EarlyStoppingMinDelta` | Gets or sets minimum improvement delta for early stopping. |
| `EarlyStoppingPatience` | Gets or sets patience for early stopping (epochs without improvement). |
| `EnsembleWeights` | Gets or sets ensemble weights (if using multiple teachers). |
| `Epochs` | Gets or sets the number of training epochs. |
| `FeatureLayerPairs` | Gets or sets layer pairs for feature-based distillation. |
| `FeatureWeight` | Gets or sets weight for feature loss (if using feature-based distillation). |
| `FreezeTeacher` | Gets or sets whether to freeze teacher model during training. |
| `LabelSmoothingFactor` | Gets or sets the label smoothing factor (if enabled). |
| `LearningRate` | Gets or sets the learning rate for student training. |
| `OnEpochComplete` | Gets or sets callback function invoked after each epoch. |
| `OutputDimension` | Gets or sets output dimension for models (if not inferrable from teacher). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SaveCheckpoints` | Gets or sets whether to save checkpoints during training. |
| `SaveOnlyBestCheckpoint` | Gets or sets whether to only save the best model checkpoint. |
| `SelfDistillationGenerations` | Gets or sets the number of self-distillation generations (if using self-distillation). |
| `Strategy` | Gets or sets the distillation strategy instance (if using custom strategy). |
| `StrategyType` | Gets or sets the distillation strategy type. |
| `Teacher` | Gets or sets the teacher model instance (if using pre-instantiated teacher). |
| `TeacherForward` | Gets or sets the teacher model forward function (alternative approach). |
| `TeacherModel` | Gets or sets the teacher IFullModel (recommended approach). |
| `TeacherModelType` | Gets or sets the type of teacher model to use. |
| `Teachers` | Gets or sets multiple teacher models (for ensemble distillation). |
| `Temperature` | Gets or sets the temperature for softmax scaling. |
| `UseEMA` | Gets or sets whether to use exponential moving average for teacher predictions (self-distillation). |
| `UseEarlyStopping` | Gets or sets whether to enable early stopping based on validation loss. |
| `UseLabelSmoothing` | Gets or sets whether to use label smoothing. |
| `ValidateAfterEpoch` | Gets or sets whether to validate model after each epoch. |
| `ValidationInputs` | Gets or sets validation data inputs (if validation is enabled). |
| `ValidationLabels` | Gets or sets validation data labels (if validation is enabled). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and throws if any are invalid. |

