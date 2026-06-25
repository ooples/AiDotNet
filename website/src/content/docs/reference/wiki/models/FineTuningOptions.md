---
title: "FineTuningOptions<T>"
description: "Configuration options for fine-tuning methods."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for fine-tuning methods.

## For Beginners

These settings control how the fine-tuning process works.
Most settings have sensible defaults based on research papers, so you can start with
the defaults and adjust as needed.

## How It Works

This class provides a comprehensive set of options that cover all fine-tuning method categories.
Each method type uses the relevant subset of options.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `Beta` | Gets or sets the beta parameter for DPO-family methods. |
| `CheckpointSteps` | Gets or sets the checkpoint frequency in steps. |
| `CompileModel` | Gets or sets whether to compile the model for faster training. |
| `ConstitutionalPrinciples` | Gets or sets the constitutional principles for CAI methods. |
| `CritiqueIterations` | Gets or sets the number of critique-revision iterations. |
| `DistillationAlpha` | Gets or sets the alpha weight between hard and soft labels. |
| `DistillationTemperature` | Gets or sets the distillation temperature. |
| `EntropyCoefficient` | Gets or sets the entropy coefficient for exploration. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GAELambda` | Gets or sets the GAE lambda for advantage estimation. |
| `GRPOGroupSize` | Gets or sets the group size for GRPO sampling. |
| `GRPOTemperature` | Gets or sets the GRPO temperature for sampling. |
| `Gamma` | Gets or sets the discount factor for rewards. |
| `GradientAccumulationSteps` | Gets or sets the gradient accumulation steps. |
| `KLCoefficient` | Gets or sets the KL coefficient for RL-based methods. |
| `KTODesirableWeight` | Gets or sets the desirable weight for KTO. |
| `KTOUndesirableWeight` | Gets or sets the undesirable weight for KTO. |
| `LabelSmoothing` | Gets or sets the label smoothing factor for preference learning. |
| `LearningRate` | Gets or sets the learning rate for fine-tuning. |
| `LoRAConfig` | Gets or sets the LoRA configuration when UseLoRA is true. |
| `LoggingSteps` | Gets or sets the logging frequency in steps. |
| `MaxCheckpoints` | Gets or sets the maximum number of checkpoints to keep. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for clipping. |
| `MaxSequenceLength` | Gets or sets the maximum sequence length. |
| `MethodType` | Gets or sets the fine-tuning method type. |
| `ORPOLambda` | Gets or sets the lambda parameter for ORPO odds ratio loss. |
| `PPOClipRange` | Gets or sets the PPO clip range. |
| `PPOEpochsPerBatch` | Gets or sets the number of PPO epochs per batch. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `RankingMargin` | Gets or sets the margin for ranking loss. |
| `RankingTemperature` | Gets or sets the temperature for ranking softmax. |
| `SPINIterations` | Gets or sets the number of self-play iterations. |
| `SimPOGamma` | Gets or sets the gamma parameter for SimPO length normalization. |
| `UseLoRA` | Gets or sets whether to use LoRA for parameter-efficient fine-tuning. |
| `UseMixedPrecision` | Gets or sets whether to use mixed precision training. |
| `ValueCoefficient` | Gets or sets the value function coefficient for PPO. |
| `WarmupRatio` | Gets or sets the warmup ratio for learning rate scheduling. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

