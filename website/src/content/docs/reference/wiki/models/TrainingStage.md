---
title: "TrainingStage<T, TInput, TOutput>"
description: "Represents a single stage in a training pipeline with comprehensive configuration options."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Options`

Represents a single stage in a training pipeline with comprehensive configuration options.

## For Beginners

Think of each stage as a chapter in a training book.
Each chapter teaches the model something different, and you can configure
exactly how that teaching happens.

## How It Works

A training stage encapsulates all configuration needed for one step in a multi-stage
training pipeline. Each stage can have its own:

## Properties

| Property | Summary |
|:-----|:--------|
| `AdamBeta1` | Gets or sets the Adam beta1 parameter (momentum). |
| `AdamBeta2` | Gets or sets the Adam beta2 parameter (RMSprop-like). |
| `AdamEpsilon` | Gets or sets the Adam epsilon for numerical stability. |
| `BatchSize` | Gets or sets the batch size for this stage. |
| `BestCheckpointMetric` | Gets or sets the metric to use for determining the best checkpoint. |
| `BestCheckpointMetricMaximize` | Gets or sets whether higher is better for the best checkpoint metric. |
| `Callbacks` | Gets or sets stage-specific callbacks. |
| `CheckpointSaveEpochs` | Gets or sets the checkpoint save interval (in epochs). |
| `CheckpointSaveSteps` | Gets or sets the checkpoint save interval (in steps). |
| `ConstitutionalPrinciples` | Gets or sets the constitutional principles for CAI stages. |
| `ContrastiveMargin` | Gets or sets the margin for contrastive preference methods. |
| `CritiqueModel` | Gets or sets the model to use for generating critiques. |
| `CritiqueRevisionRounds` | Gets or sets the number of critique-revision rounds. |
| `CustomEvaluationFunction` | Gets or sets the custom evaluation function for this stage. |
| `CustomLossFunction` | Gets or sets custom loss function for this stage. |
| `CustomMetricName` | Gets or sets custom metric name when BestCheckpointMetric is Custom. |
| `CustomTrainingFunction` | Gets or sets the custom training function for custom stages. |
| `DPOBeta` | Gets or sets the beta parameter for DPO/IPO loss. |
| `DataMixingRatios` | Gets or sets the data mixing ratio when combining multiple datasets. |
| `DataPreprocessor` | Gets or sets custom data preprocessing for this stage. |
| `DataShuffleSeed` | Gets or sets the random seed for data shuffling (for reproducibility). |
| `Description` | Gets or sets a description of what this stage accomplishes. |
| `DistillationAlpha` | Gets or sets the alpha for balancing hard vs soft targets. |
| `DistillationLayerMapping` | Gets or sets the layer mapping for intermediate distillation. |
| `DistillationTemperature` | Gets or sets the distillation temperature. |
| `DistributedStrategy` | Gets or sets the distributed training strategy for this stage. |
| `EarlyStopping` | Gets or sets early stopping configuration specific to this stage. |
| `Enabled` | Gets or sets whether this stage is enabled (skipped if false). |
| `EntropyCoefficient` | Gets or sets the entropy bonus coefficient for exploration. |
| `Epochs` | Gets or sets the number of epochs for this stage. |
| `EvaluationSteps` | Gets or sets the evaluation interval (in steps). |
| `FineTuningMethod` | Gets or sets the fine-tuning method to use in this stage. |
| `FreezeBaseModel` | Gets or sets whether to freeze the base model during this stage. |
| `FrozenLayers` | Gets or sets layer names/patterns to freeze during this stage. |
| `GAELambda` | Gets or sets the GAE lambda for advantage estimation. |
| `GRPOGroupSize` | Gets or sets the group size for GRPO. |
| `GRPOUseRelativeRewards` | Gets or sets whether to use relative rewards in GRPO. |
| `GradientAccumulationSteps` | Gets or sets the gradient accumulation steps. |
| `GradualUnfreezingInterval` | Gets or sets the epoch interval for gradual unfreezing. |
| `InitialLossScale` | Gets or sets the initial loss scale for mixed precision. |
| `IsEvaluationOnly` | Gets or sets whether this stage is evaluation-only (no training). |
| `KLPenaltyCoefficient` | Gets or sets the KL penalty coefficient for RLHF. |
| `LearningRate` | Gets or sets the learning rate for this stage. |
| `LoRAAlpha` | Gets or sets the LoRA alpha scaling factor. |
| `LoRADropout` | Gets or sets the LoRA dropout rate. |
| `LoRARank` | Gets or sets the LoRA rank (dimension of low-rank matrices). |
| `LoRATargetModules` | Gets or sets which modules to apply LoRA to. |
| `LogGradientNorms` | Gets or sets whether to log gradient norms. |
| `LogLearningRate` | Gets or sets whether to log learning rate. |
| `LoggingSteps` | Gets or sets the logging interval (in steps). |
| `MaxCheckpointsToKeep` | Gets or sets the maximum number of checkpoints to keep. |
| `MaxDuration` | Gets or sets the maximum duration for this stage. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for gradient clipping. |
| `MaxSteps` | Gets or sets the maximum number of steps for this stage. |
| `MaxTrainingSamples` | Gets or sets the maximum number of training samples to use. |
| `MergeLoRAAfterTraining` | Gets or sets whether to merge LoRA weights into base model after training. |
| `Metadata` | Gets or sets custom metadata for this stage. |
| `MetricsToTrack` | Gets or sets the metrics to track during this stage. |
| `MinLearningRate` | Gets or sets the minimum learning rate (for schedulers with decay). |
| `MixedPrecisionDType` | Gets or sets the mixed precision data type. |
| `Name` | Gets or sets the name of this stage for logging and identification. |
| `NumCycles` | Gets or sets the number of cycles for cosine scheduler with restarts. |
| `OptimizerOverride` | Gets or sets the optimizer type override for this stage. |
| `Options` | Gets or sets the fine-tuning options for this stage. |
| `PPOClipRange` | Gets or sets the PPO clip range. |
| `PPOEpochsPerBatch` | Gets or sets the number of PPO epochs per batch. |
| `PreferenceLabelSmoothing` | Gets or sets the label smoothing factor for preference learning. |
| `PreferenceLossType` | Gets or sets the loss type for preference optimization. |
| `QLoRABits` | Gets or sets the quantization bits for QLoRA. |
| `RandomSeed` | Gets or sets the random seed for this stage. |
| `ReferenceModelUpdateInterval` | Gets or sets the interval (in steps) for updating the reference model. |
| `RejectionSamplingMinReward` | Gets or sets the minimum reward threshold for rejection sampling. |
| `RejectionSamplingN` | Gets or sets the number of samples to generate for rejection sampling. |
| `RejectionSamplingTopK` | Gets or sets the top-K samples to keep from rejection sampling. |
| `RewardModel` | Gets or sets the reward model to use for RLHF stages. |
| `RolloutSamples` | Gets or sets the number of rollout samples per update. |
| `RunCondition` | Gets or sets conditions that must be met to run this stage. |
| `SaveCheckpointAfter` | Gets or sets whether to save a checkpoint after this stage. |
| `SaveOnlyBest` | Gets or sets whether to save only the best checkpoint based on validation metrics. |
| `SchedulerPower` | Gets or sets the power for polynomial decay scheduler. |
| `SchedulerType` | Gets or sets the learning rate scheduler type. |
| `SelfPlayIterations` | Gets or sets the number of self-play iterations. |
| `SelfPlayResponsesPerPrompt` | Gets or sets the number of responses to generate per prompt in self-play. |
| `SelfPlayTemperature` | Gets or sets the generation temperature for self-play responses. |
| `ShareReferenceModel` | Gets or sets whether to share reference model with the training model. |
| `ShuffleData` | Gets or sets whether to shuffle the training data each epoch. |
| `StageType` | Gets or sets the type of training stage. |
| `SyncBatchNorm` | Gets or sets whether to sync batch normalization across devices. |
| `Tags` | Gets or sets tags for categorizing this stage. |
| `TeacherModel` | Gets or sets the teacher model for distillation stages. |
| `TrainableLayers` | Gets or sets layer names/patterns to unfreeze (train) during this stage. |
| `TrainingData` | Gets or sets the training data for this stage. |
| `UnfreezeTopNLayers` | Gets or sets the number of layers to unfreeze from the top. |
| `UpdateReferenceModel` | Gets or sets whether to update the reference model periodically. |
| `UseDeterministicAlgorithms` | Gets or sets whether to use deterministic algorithms (may be slower). |
| `UseDynamicLossScaling` | Gets or sets whether to use dynamic loss scaling for mixed precision. |
| `UseGradientCheckpointing` | Gets or sets whether to use gradient checkpointing to save memory. |
| `UseGradualUnfreezing` | Gets or sets whether to gradually unfreeze layers during training. |
| `UseIntermediateDistillation` | Gets or sets whether to use intermediate layer distillation. |
| `UseLoRA` | Gets or sets whether to use LoRA (Low-Rank Adaptation) for this stage. |
| `UseMixedPrecision` | Gets or sets whether to use mixed precision training (FP16/BF16). |
| `UseQLoRA` | Gets or sets whether to use QLoRA (quantized LoRA) for memory efficiency. |
| `UseReferenceModel` | Gets or sets whether to use a reference model for preference methods. |
| `ValidationData` | Gets or sets the validation data for this stage. |
| `ValueFunctionCoefficient` | Gets or sets the value function coefficient for PPO. |
| `WarmupRatio` | Gets or sets the warmup ratio (fraction of total steps for warmup). |
| `WarmupSteps` | Gets or sets the number of warmup steps. |
| `WeightDecay` | Gets or sets the weight decay (L2 regularization) coefficient. |

