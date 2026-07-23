using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;

namespace AiDotNet.Models.Options;

/// <summary>
/// Represents a single stage in a training pipeline with comprehensive configuration options.
/// </summary>
/// <remarks>
/// <para>
/// A training stage encapsulates all configuration needed for one step in a multi-stage
/// training pipeline. Each stage can have its own:
/// </para>
/// <list type="bullet">
/// <item><description>Training method (SFT, DPO, RLHF, etc.)</description></item>
/// <item><description>Optimizer and learning rate</description></item>
/// <item><description>Dataset and validation data</description></item>
/// <item><description>Layer freezing and LoRA configuration</description></item>
/// <item><description>Scheduler and warmup settings</description></item>
/// <item><description>Early stopping and checkpointing</description></item>
/// </list>
/// <para><b>For Beginners:</b> Think of each stage as a chapter in a training book.
/// Each chapter teaches the model something different, and you can configure
/// exactly how that teaching happens.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class TrainingStage<T, TInput, TOutput>
{
    // ========================================================================
    // Basic Stage Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the name of this stage for logging and identification.
    /// </summary>
    public string Name { get; set; } = "Training Stage";

    /// <summary>
    /// Gets or sets a description of what this stage accomplishes.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the type of training stage.
    /// </summary>
    public TrainingStageType StageType { get; set; } = TrainingStageType.SupervisedFineTuning;

    /// <summary>
    /// Gets or sets the fine-tuning method to use in this stage.
    /// </summary>
    public FineTuningMethodType FineTuningMethod { get; set; } = FineTuningMethodType.SFT;

    /// <summary>
    /// Gets or sets whether this stage is enabled (skipped if false).
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets conditions that must be met to run this stage.
    /// </summary>
    /// <remarks>
    /// If the condition returns false, the stage is skipped.
    /// Receives the result of the previous stage (null for first stage).
    /// </remarks>
    public Func<TrainingStageResult<T, TInput, TOutput>?, bool>? RunCondition { get; set; }

    /// <summary>
    /// Gets or sets whether this stage is evaluation-only (no training).
    /// </summary>
    public bool IsEvaluationOnly { get; set; }

    // ========================================================================
    // Data Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the training data for this stage.
    /// </summary>
    public FineTuningData<T, TInput, TOutput>? TrainingData { get; set; }

    /// <summary>
    /// Gets or sets the validation data for this stage.
    /// </summary>
    public FineTuningData<T, TInput, TOutput>? ValidationData { get; set; }

    /// <summary>
    /// Gets or sets the data mixing ratio when combining multiple datasets.
    /// </summary>
    /// <remarks>
    /// Keys are dataset names/identifiers, values are sampling weights.
    /// For example: { "instruction": 0.5, "conversation": 0.3, "safety": 0.2 }
    /// </remarks>
    public Dictionary<string, double>? DataMixingRatios { get; set; }

    /// <summary>
    /// Gets or sets whether to shuffle the training data each epoch.
    /// </summary>
    public bool ShuffleData { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for data shuffling (for reproducibility).
    /// </summary>
    public int? DataShuffleSeed { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of training samples to use.
    /// </summary>
    /// <remarks>
    /// Useful for limiting data in curriculum learning or quick experiments.
    /// </remarks>
    public int? MaxTrainingSamples { get; set; }

    // ========================================================================
    // Training Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the fine-tuning options for this stage.
    /// </summary>
    public FineTuningOptions<T>? Options { get; set; }

    /// <summary>
    /// Gets or sets the number of epochs for this stage.
    /// </summary>
    /// <remarks>
    /// Overrides the value in Options if set. Default is 3 epochs.
    /// </remarks>
    public int Epochs { get; set; } = 3;

    /// <summary>
    /// Gets or sets the batch size for this stage.
    /// </summary>
    /// <remarks>
    /// Overrides the value in Options if set. Default is 8.
    /// </remarks>
    public int BatchSize { get; set; } = 8;

    /// <summary>
    /// Gets or sets the gradient accumulation steps.
    /// </summary>
    /// <remarks>
    /// Allows effective larger batch sizes with limited memory.
    /// Effective batch size = BatchSize * GradientAccumulationSteps
    /// </remarks>
    public int GradientAccumulationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to use gradient checkpointing to save memory.
    /// </summary>
    public bool UseGradientCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <remarks>
    /// Default is 1.0. Set to 0 to disable gradient clipping.
    /// </remarks>
    public double MaxGradientNorm { get; set; } = 1.0;

    // ========================================================================
    // Optimizer Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the optimizer type override for this stage.
    /// </summary>
    /// <remarks>
    /// Default is AdamW, the standard choice for fine-tuning.
    /// </remarks>
    public OptimizerType OptimizerOverride { get; set; } = OptimizerType.AdamW;

    /// <summary>
    /// Gets or sets the learning rate for this stage.
    /// </summary>
    /// <remarks>
    /// Default is 2e-5, a common choice for fine-tuning pre-trained models.
    /// </remarks>
    public double LearningRate { get; set; } = 2e-5;

    /// <summary>
    /// Gets or sets the minimum learning rate (for schedulers with decay).
    /// </summary>
    /// <remarks>
    /// Default is 0 (learning rate can decay to zero).
    /// </remarks>
    public double MinLearningRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the weight decay (L2 regularization) coefficient.
    /// </summary>
    /// <remarks>
    /// Default is 0.01, standard for AdamW.
    /// </remarks>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the Adam beta1 parameter (momentum).
    /// </summary>
    /// <remarks>
    /// Default is 0.9, standard for Adam/AdamW.
    /// </remarks>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the Adam beta2 parameter (RMSprop-like).
    /// </summary>
    /// <remarks>
    /// Default is 0.999, standard for Adam/AdamW.
    /// </remarks>
    public double AdamBeta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets the Adam epsilon for numerical stability.
    /// </summary>
    /// <remarks>
    /// Default is 1e-8, standard for Adam/AdamW.
    /// </remarks>
    public double AdamEpsilon { get; set; } = 1e-8;

    // ========================================================================
    // Learning Rate Scheduler Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the learning rate scheduler type.
    /// </summary>
    /// <remarks>
    /// Default is CosineAnnealing, which works well for most fine-tuning scenarios.
    /// </remarks>
    public LearningRateSchedulerType SchedulerType { get; set; } = LearningRateSchedulerType.CosineAnnealing;

    /// <summary>
    /// Gets or sets the number of warmup steps.
    /// </summary>
    /// <remarks>
    /// Default is 0. Set this or WarmupRatio, not both.
    /// </remarks>
    public int WarmupSteps { get; set; } = 0;

    /// <summary>
    /// Gets or sets the warmup ratio (fraction of total steps for warmup).
    /// </summary>
    /// <remarks>
    /// Default is 0.1 (10% of training for warmup).
    /// If WarmupSteps is set, this is ignored.
    /// </remarks>
    public double WarmupRatio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of cycles for cosine scheduler with restarts.
    /// </summary>
    /// <remarks>
    /// Default is 1 (no restarts, single decay to min learning rate).
    /// </remarks>
    public int NumCycles { get; set; } = 1;

    /// <summary>
    /// Gets or sets the power for polynomial decay scheduler.
    /// </summary>
    /// <remarks>
    /// Default is 1.0 (linear decay). Higher values = faster initial decay.
    /// </remarks>
    public double SchedulerPower { get; set; } = 1.0;

    // ========================================================================
    // Layer Freezing Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to freeze the base model during this stage.
    /// </summary>
    /// <remarks>
    /// Default is false (train all layers). Set to true for LoRA or when using adapters.
    /// </remarks>
    public bool FreezeBaseModel { get; set; } = false;

    /// <summary>
    /// Gets or sets layer names/patterns to freeze during this stage.
    /// </summary>
    /// <remarks>
    /// Supports patterns like "encoder.*", "layer.0-5.*", "embedding".
    /// Default is empty (no specific layers frozen unless FreezeBaseModel is true).
    /// </remarks>
    public string[] FrozenLayers { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets layer names/patterns to unfreeze (train) during this stage.
    /// </summary>
    /// <remarks>
    /// If FreezeBaseModel is true, only these layers will be trained.
    /// Supports patterns like "classifier", "lm_head", "layer.10-11.*".
    /// Default is empty (all unfrozen layers are trainable).
    /// </remarks>
    public string[] TrainableLayers { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the number of layers to unfreeze from the top.
    /// </summary>
    /// <remarks>
    /// Default is 0 (use FrozenLayers/TrainableLayers patterns instead).
    /// Common approach: freeze most layers, train only top N layers.
    /// </remarks>
    public int UnfreezeTopNLayers { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to gradually unfreeze layers during training.
    /// </summary>
    /// <remarks>
    /// Default is false. When true, layers are unfrozen progressively during training.
    /// </remarks>
    public bool UseGradualUnfreezing { get; set; } = false;

    /// <summary>
    /// Gets or sets the epoch interval for gradual unfreezing.
    /// </summary>
    /// <remarks>
    /// Default is 1 epoch. Every N epochs, unfreeze one more layer group.
    /// </remarks>
    public int GradualUnfreezingInterval { get; set; } = 1;

    // ========================================================================
    // LoRA / PEFT Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to use LoRA (Low-Rank Adaptation) for this stage.
    /// </summary>
    /// <remarks>
    /// Default is false (full fine-tuning). Set to true for parameter-efficient training.
    /// </remarks>
    public bool UseLoRA { get; set; } = false;

    /// <summary>
    /// Gets or sets the LoRA rank (dimension of low-rank matrices).
    /// </summary>
    /// <remarks>
    /// Default is 16. Common values: 4, 8, 16, 32, 64. Higher = more capacity but more parameters.
    /// </remarks>
    public int LoRARank { get; set; } = 16;

    /// <summary>
    /// Gets or sets the LoRA alpha scaling factor.
    /// </summary>
    /// <remarks>
    /// Default is 32 (2x rank). Effective scaling = alpha / rank.
    /// </remarks>
    public double LoRAAlpha { get; set; } = 32.0;

    /// <summary>
    /// Gets or sets the LoRA dropout rate.
    /// </summary>
    /// <remarks>
    /// Default is 0.05 (5%). Light dropout helps regularization.
    /// </remarks>
    public double LoRADropout { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets which modules to apply LoRA to.
    /// </summary>
    /// <remarks>
    /// Default targets query and value projections.
    /// Common patterns: ["q_proj", "v_proj"], ["query", "key", "value", "output"].
    /// </remarks>
    public string[] LoRATargetModules { get; set; } = new[] { "q_proj", "v_proj" };

    /// <summary>
    /// Gets or sets whether to use QLoRA (quantized LoRA) for memory efficiency.
    /// </summary>
    /// <remarks>
    /// Default is false. Set to true for 4-bit or 8-bit quantized training.
    /// </remarks>
    public bool UseQLoRA { get; set; } = false;

    /// <summary>
    /// Gets or sets the quantization bits for QLoRA.
    /// </summary>
    /// <remarks>
    /// Default is 4 bits (most memory efficient). Use 8 for higher precision.
    /// </remarks>
    public int QLoRABits { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to merge LoRA weights into base model after training.
    /// </summary>
    /// <remarks>
    /// Default is false. Set to true to produce a merged model for deployment.
    /// </remarks>
    public bool MergeLoRAAfterTraining { get; set; } = false;

    // ========================================================================
    // Reference Model Configuration (for Preference Methods)
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to use a reference model for preference methods.
    /// </summary>
    /// <remarks>
    /// Required for DPO, IPO, etc. Not required for SimPO, ORPO.
    /// Default is true (use reference model for KL constraint).
    /// </remarks>
    public bool UseReferenceModel { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to share reference model with the training model.
    /// </summary>
    /// <remarks>
    /// Default is true (memory efficient). If false, loads a separate copy.
    /// </remarks>
    public bool ShareReferenceModel { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to update the reference model periodically.
    /// </summary>
    /// <remarks>
    /// Default is false (frozen reference model).
    /// </remarks>
    public bool UpdateReferenceModel { get; set; } = false;

    /// <summary>
    /// Gets or sets the interval (in steps) for updating the reference model.
    /// </summary>
    /// <remarks>
    /// Default is 100 steps. Only used when UpdateReferenceModel is true.
    /// </remarks>
    public int ReferenceModelUpdateInterval { get; set; } = 100;

    // ========================================================================
    // Preference/DPO-Specific Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the beta parameter for DPO/IPO loss.
    /// </summary>
    /// <remarks>
    /// Default is 0.1. Controls the strength of the KL constraint. Typical range: 0.01-0.5.
    /// </remarks>
    public double DPOBeta { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the label smoothing factor for preference learning.
    /// </summary>
    /// <remarks>
    /// Default is 0.0 (no smoothing). Values like 0.1 can help with noisy preferences.
    /// </remarks>
    public double PreferenceLabelSmoothing { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the loss type for preference optimization.
    /// </summary>
    /// <remarks>
    /// Default is Sigmoid (standard DPO).
    /// </remarks>
    public PreferenceLossType PreferenceLossType { get; set; } = PreferenceLossType.Sigmoid;

    /// <summary>
    /// Gets or sets the margin for contrastive preference methods.
    /// </summary>
    /// <remarks>
    /// Default is 0.0. Used by contrastive methods like CPO.
    /// </remarks>
    public double ContrastiveMargin { get; set; } = 0.0;

    // ========================================================================
    // RLHF/PPO-Specific Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the reward model to use for RLHF stages.
    /// </summary>
    /// <remarks>
    /// Required for PPO/RLHF. Can be null if using a reward-free method like DPO.
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? RewardModel { get; set; }

    /// <summary>
    /// Gets or sets the KL penalty coefficient for RLHF.
    /// </summary>
    /// <remarks>
    /// Default is 0.01. Prevents the policy from diverging too far from the reference.
    /// </remarks>
    public double KLPenaltyCoefficient { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the PPO clip range.
    /// </summary>
    /// <remarks>
    /// Default is 0.2. Limits how much the policy can change per update.
    /// </remarks>
    public double PPOClipRange { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the value function coefficient for PPO.
    /// </summary>
    /// <remarks>
    /// Default is 0.5. Weight of value loss relative to policy loss.
    /// </remarks>
    public double ValueFunctionCoefficient { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the entropy bonus coefficient for exploration.
    /// </summary>
    /// <remarks>
    /// Default is 0.01. Encourages exploration by penalizing deterministic policies.
    /// </remarks>
    public double EntropyCoefficient { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the GAE lambda for advantage estimation.
    /// </summary>
    /// <remarks>
    /// Default is 0.95. Controls bias-variance tradeoff in advantage estimation.
    /// </remarks>
    public double GAELambda { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the number of PPO epochs per batch.
    /// </summary>
    /// <remarks>
    /// Default is 4. Number of times to reuse collected experiences.
    /// </remarks>
    public int PPOEpochsPerBatch { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of rollout samples per update.
    /// </summary>
    /// <remarks>
    /// Default is 2048. Number of environment steps to collect before each PPO update.
    /// </remarks>
    public int RolloutSamples { get; set; } = 2048;

    // ========================================================================
    // GRPO-Specific Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the group size for GRPO.
    /// </summary>
    /// <remarks>
    /// Default is 4. Number of responses to generate per prompt for group ranking.
    /// </remarks>
    public int GRPOGroupSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to use relative rewards in GRPO.
    /// </summary>
    /// <remarks>
    /// Default is true. Normalize rewards within each group for stable training.
    /// </remarks>
    public bool GRPOUseRelativeRewards { get; set; } = true;

    // ========================================================================
    // Constitutional AI Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the constitutional principles for CAI stages.
    /// </summary>
    /// <remarks>
    /// Default includes standard HHH (Helpful, Harmless, Honest) principles.
    /// </remarks>
    public string[] ConstitutionalPrinciples { get; set; } = new[]
    {
        "Please choose the response that is most helpful, accurate, and harmless.",
        "Please choose the response that is honest and does not make up information.",
        "Please choose the response that is less toxic, rude, or harmful.",
        "Please choose the response that respects user privacy and autonomy.",
        "Please choose the response that is more ethical and less biased."
    };

    /// <summary>
    /// Gets or sets the number of critique-revision rounds.
    /// </summary>
    /// <remarks>
    /// Default is 2. More rounds = more refined responses but slower training.
    /// </remarks>
    public int CritiqueRevisionRounds { get; set; } = 2;

    /// <summary>
    /// Gets or sets the model to use for generating critiques.
    /// </summary>
    /// <remarks>
    /// If null, uses the model being trained for self-critique.
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? CritiqueModel { get; set; }

    // ========================================================================
    // Self-Play Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the number of self-play iterations.
    /// </summary>
    /// <remarks>
    /// Default is 3. Number of self-play rounds per training cycle.
    /// </remarks>
    public int SelfPlayIterations { get; set; } = 3;

    /// <summary>
    /// Gets or sets the generation temperature for self-play responses.
    /// </summary>
    /// <remarks>
    /// Default is 0.7. Higher values = more diverse responses, lower = more focused.
    /// </remarks>
    public double SelfPlayTemperature { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the number of responses to generate per prompt in self-play.
    /// </summary>
    /// <remarks>
    /// Default is 4. More responses = better coverage but slower training.
    /// </remarks>
    public int SelfPlayResponsesPerPrompt { get; set; } = 4;

    // ========================================================================
    // Rejection Sampling Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the number of samples to generate for rejection sampling.
    /// </summary>
    /// <remarks>
    /// Default is 10. Generate N responses and select the best ones.
    /// </remarks>
    public int RejectionSamplingN { get; set; } = 10;

    /// <summary>
    /// Gets or sets the top-K samples to keep from rejection sampling.
    /// </summary>
    /// <remarks>
    /// Default is 1. Keep only the best response per prompt.
    /// </remarks>
    public int RejectionSamplingTopK { get; set; } = 1;

    /// <summary>
    /// Gets or sets the minimum reward threshold for rejection sampling.
    /// </summary>
    /// <remarks>
    /// Default is 0.0. Only keep responses with reward above this threshold.
    /// </remarks>
    public double RejectionSamplingMinReward { get; set; } = 0.0;

    // ========================================================================
    // Knowledge Distillation Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the teacher model for distillation stages.
    /// </summary>
    /// <remarks>
    /// Required for knowledge distillation. The larger model to distill from.
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? TeacherModel { get; set; }

    /// <summary>
    /// Gets or sets the distillation temperature.
    /// </summary>
    /// <remarks>
    /// Default is 2.0. Higher values produce softer probability distributions.
    /// </remarks>
    public double DistillationTemperature { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the alpha for balancing hard vs soft targets.
    /// </summary>
    /// <remarks>
    /// Default is 0.5. alpha * soft_loss + (1-alpha) * hard_loss.
    /// </remarks>
    public double DistillationAlpha { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use intermediate layer distillation.
    /// </summary>
    /// <remarks>
    /// Default is false. When true, also distills intermediate representations.
    /// </remarks>
    public bool UseIntermediateDistillation { get; set; } = false;

    /// <summary>
    /// Gets or sets the layer mapping for intermediate distillation.
    /// </summary>
    /// <remarks>
    /// Maps teacher layers to student layers: { teacherLayerIdx: studentLayerIdx }.
    /// Default is empty (auto-map if UseIntermediateDistillation is true).
    /// </remarks>
    public Dictionary<int, int> DistillationLayerMapping { get; set; } = new Dictionary<int, int>();

    // ========================================================================
    // Mixed Precision Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to use mixed precision training (FP16/BF16).
    /// </summary>
    /// <remarks>
    /// Default is false (FP32). Enable for faster training with lower memory.
    /// </remarks>
    public bool UseMixedPrecision { get; set; } = false;

    /// <summary>
    /// Gets or sets the mixed precision data type.
    /// </summary>
    /// <remarks>
    /// Default is FP16 for broad compatibility. BF16 is better on Ampere+ GPUs.
    /// </remarks>
    public MixedPrecisionType MixedPrecisionDType { get; set; } = MixedPrecisionType.FP16;

    /// <summary>
    /// Gets or sets whether to use dynamic loss scaling for mixed precision.
    /// </summary>
    /// <remarks>
    /// Default is true. Required for FP16, optional for BF16.
    /// </remarks>
    public bool UseDynamicLossScaling { get; set; } = true;

    /// <summary>
    /// Gets or sets the initial loss scale for mixed precision.
    /// </summary>
    /// <remarks>
    /// Default is 65536.0. Starting scale for dynamic loss scaling.
    /// </remarks>
    public double InitialLossScale { get; set; } = 65536.0;

    // ========================================================================
    // Checkpointing and Saving Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to save a checkpoint after this stage.
    /// </summary>
    /// <remarks>
    /// Default is true. Always save after each stage for recovery.
    /// </remarks>
    public bool SaveCheckpointAfter { get; set; } = true;

    /// <summary>
    /// Gets or sets the checkpoint save interval (in steps).
    /// </summary>
    /// <remarks>
    /// Default is 500 steps. Set to 0 to only save at epoch boundaries.
    /// </remarks>
    public int CheckpointSaveSteps { get; set; } = 500;

    /// <summary>
    /// Gets or sets the checkpoint save interval (in epochs).
    /// </summary>
    /// <remarks>
    /// Default is 1 (save every epoch). Set to 0 to use step-based saving only.
    /// </remarks>
    public int CheckpointSaveEpochs { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum number of checkpoints to keep.
    /// </summary>
    /// <remarks>
    /// Default is 3. Older checkpoints are deleted to save disk space.
    /// </remarks>
    public int MaxCheckpointsToKeep { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to save only the best checkpoint based on validation metrics.
    /// </summary>
    /// <remarks>
    /// Default is false. When true, only keeps the checkpoint with best metric.
    /// </remarks>
    public bool SaveOnlyBest { get; set; } = false;

    /// <summary>
    /// Gets or sets the metric to use for determining the best checkpoint.
    /// </summary>
    /// <remarks>
    /// Default is Loss (lower is better).
    /// </remarks>
    public CheckpointMetricType BestCheckpointMetric { get; set; } = CheckpointMetricType.Loss;

    /// <summary>
    /// Gets or sets custom metric name when BestCheckpointMetric is Custom.
    /// </summary>
    /// <remarks>
    /// Only used when BestCheckpointMetric is set to Custom.
    /// </remarks>
    public string CustomMetricName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether higher is better for the best checkpoint metric.
    /// </summary>
    /// <remarks>
    /// Default is false (lower is better, appropriate for Loss).
    /// Set to true for metrics like Accuracy, F1, BLEU, etc.
    /// </remarks>
    public bool BestCheckpointMetricMaximize { get; set; } = false;

    // ========================================================================
    // Early Stopping Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets early stopping configuration specific to this stage.
    /// </summary>
    /// <remarks>
    /// Default configuration with patience=5 and monitoring loss.
    /// </remarks>
    public EarlyStoppingConfig EarlyStopping { get; set; } = new EarlyStoppingConfig();

    /// <summary>
    /// Gets or sets the maximum duration for this stage.
    /// </summary>
    /// <remarks>
    /// Default is 24 hours. Set to TimeSpan.MaxValue for no limit.
    /// </remarks>
    public TimeSpan MaxDuration { get; set; } = TimeSpan.FromHours(24);

    /// <summary>
    /// Gets or sets the maximum number of steps for this stage.
    /// </summary>
    /// <remarks>
    /// Default is 0 (no step limit, use Epochs instead).
    /// </remarks>
    public int MaxSteps { get; set; } = 0;

    // ========================================================================
    // Logging and Monitoring Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the logging interval (in steps).
    /// </summary>
    /// <remarks>
    /// Default is 10 steps. Log training metrics every N steps.
    /// </remarks>
    public int LoggingSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the evaluation interval (in steps).
    /// </summary>
    /// <remarks>
    /// Default is 100 steps. Run validation every N steps.
    /// </remarks>
    public int EvaluationSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the metrics to track during this stage.
    /// </summary>
    /// <remarks>
    /// Default includes loss and perplexity.
    /// </remarks>
    public string[] MetricsToTrack { get; set; } = new[] { "loss", "perplexity" };

    /// <summary>
    /// Gets or sets whether to log gradient norms.
    /// </summary>
    /// <remarks>
    /// Default is false. Enable to debug gradient issues.
    /// </remarks>
    public bool LogGradientNorms { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to log learning rate.
    /// </summary>
    /// <remarks>
    /// Default is true. Useful for verifying scheduler behavior.
    /// </remarks>
    public bool LogLearningRate { get; set; } = true;

    // ========================================================================
    // Callbacks and Custom Logic
    // ========================================================================

    /// <summary>
    /// Gets or sets stage-specific callbacks.
    /// </summary>
    public StageCallbacks<T, TInput, TOutput>? Callbacks { get; set; }

    /// <summary>
    /// Gets or sets the custom training function for custom stages.
    /// </summary>
    public Func<IFullModel<T, TInput, TOutput>, FineTuningData<T, TInput, TOutput>, CancellationToken, Task<IFullModel<T, TInput, TOutput>>>? CustomTrainingFunction { get; set; }

    /// <summary>
    /// Gets or sets the custom evaluation function for this stage.
    /// </summary>
    public Func<IFullModel<T, TInput, TOutput>, FineTuningData<T, TInput, TOutput>?, Task<Dictionary<string, double>>>? CustomEvaluationFunction { get; set; }

    /// <summary>
    /// Gets or sets custom data preprocessing for this stage.
    /// </summary>
    public Func<FineTuningData<T, TInput, TOutput>, FineTuningData<T, TInput, TOutput>>? DataPreprocessor { get; set; }

    /// <summary>
    /// Gets or sets custom loss function for this stage.
    /// </summary>
    public ILossFunction<T>? CustomLossFunction { get; set; }

    // ========================================================================
    // Reproducibility Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the random seed for this stage.
    /// </summary>
    /// <remarks>
    /// Default is 42. Set to different values for different runs.
    /// </remarks>
    public int RandomSeed { get; set; } = 42;

    /// <summary>
    /// Gets or sets whether to use deterministic algorithms (may be slower).
    /// </summary>
    /// <remarks>
    /// Default is false. Enable for exact reproducibility at cost of speed.
    /// </remarks>
    public bool UseDeterministicAlgorithms { get; set; } = false;

    // ========================================================================
    // Distributed Training Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the distributed training strategy for this stage.
    /// </summary>
    /// <remarks>
    /// Default is None (single device). Use DataParallel or FSDP for multi-GPU.
    /// </remarks>
    public DistributedStrategy DistributedStrategy { get; set; } = DistributedStrategy.None;

    /// <summary>
    /// Gets or sets whether to sync batch normalization across devices.
    /// </summary>
    /// <remarks>
    /// Default is false. Enable for better accuracy in distributed training.
    /// </remarks>
    public bool SyncBatchNorm { get; set; } = false;

    // ========================================================================
    // Metadata
    // ========================================================================

    /// <summary>
    /// Gets or sets custom metadata for this stage.
    /// </summary>
    /// <remarks>
    /// Empty by default. Use to store custom key-value pairs.
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Gets or sets tags for categorizing this stage.
    /// </summary>
    /// <remarks>
    /// Empty by default. Useful for filtering and organizing stages.
    /// </remarks>
    public string[] Tags { get; set; } = Array.Empty<string>();
}
