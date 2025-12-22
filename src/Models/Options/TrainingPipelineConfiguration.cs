using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for a multi-step training pipeline with customizable stages.
/// </summary>
/// <remarks>
/// <para>
/// A training pipeline defines a sequence of training stages that are executed in order.
/// Each stage can have its own training method, optimizer, learning rate, dataset, and
/// evaluation criteria. This enables advanced training workflows like:
/// </para>
/// <list type="bullet">
/// <item><description><b>InstructGPT:</b> SFT → Reward Model → PPO</description></item>
/// <item><description><b>Llama 2:</b> SFT → Rejection Sampling → DPO</description></item>
/// <item><description><b>Anthropic Claude:</b> SFT → Constitutional AI → RLHF</description></item>
/// <item><description><b>DeepSeek:</b> SFT → GRPO</description></item>
/// <item><description><b>Curriculum:</b> Easy → Medium → Hard stages</description></item>
/// </list>
/// <para><b>For Beginners:</b> Think of this as a recipe with multiple cooking steps.
/// Just like you might marinate, then sear, then bake - training can have multiple
/// phases where each phase teaches the model something different.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class TrainingPipelineConfiguration<T, TInput, TOutput>
{
    // ========================================================================
    // Core Pipeline Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the name of this pipeline for identification.
    /// </summary>
    public string Name { get; set; } = "Training Pipeline";

    /// <summary>
    /// Gets or sets the description of this pipeline.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the ordered list of training stages in the pipeline.
    /// </summary>
    /// <remarks>
    /// Stages are executed sequentially. The output model from each stage
    /// becomes the input model for the next stage.
    /// </remarks>
    public List<TrainingStage<T, TInput, TOutput>> Stages { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to use automatic pipeline selection when no stages are defined.
    /// </summary>
    /// <remarks>
    /// When true and Stages is empty, the system analyzes available data and
    /// automatically constructs an appropriate training pipeline.
    /// </remarks>
    public bool EnableAutoSelection { get; set; } = true;

    // ========================================================================
    // Global Training Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the global random seed for reproducibility across all stages.
    /// </summary>
    /// <remarks>
    /// Default is 42. Use for reproducible training runs.
    /// </remarks>
    public int GlobalSeed { get; set; } = 42;

    /// <summary>
    /// Gets or sets the default optimizer type for all stages.
    /// </summary>
    /// <remarks>
    /// Default is AdamW. Individual stages can override this with OptimizerOverride.
    /// </remarks>
    public OptimizerType DefaultOptimizer { get; set; } = OptimizerType.AdamW;

    /// <summary>
    /// Gets or sets the default learning rate for all stages.
    /// </summary>
    /// <remarks>
    /// Default is 2e-5, standard for fine-tuning pre-trained models.
    /// </remarks>
    public double DefaultLearningRate { get; set; } = 2e-5;

    /// <summary>
    /// Gets or sets the default batch size for all stages.
    /// </summary>
    /// <remarks>
    /// Default is 8. Adjust based on available GPU memory.
    /// </remarks>
    public int DefaultBatchSize { get; set; } = 8;

    /// <summary>
    /// Gets or sets whether to use mixed precision training globally.
    /// </summary>
    /// <remarks>
    /// Default is false. Enable for faster training with lower memory.
    /// </remarks>
    public bool UseMixedPrecision { get; set; } = false;

    /// <summary>
    /// Gets or sets the mixed precision data type.
    /// </summary>
    /// <remarks>
    /// Default is FP16 for broad GPU compatibility. Use BF16 on Ampere+ GPUs.
    /// </remarks>
    public MixedPrecisionType MixedPrecisionDType { get; set; } = MixedPrecisionType.FP16;

    /// <summary>
    /// Gets or sets whether to use gradient checkpointing globally.
    /// </summary>
    /// <remarks>
    /// Default is false. Enable to reduce memory at cost of speed.
    /// </remarks>
    public bool UseGradientCheckpointing { get; set; } = false;

    // ========================================================================
    // Checkpointing Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to save checkpoints between stages.
    /// </summary>
    /// <remarks>
    /// Default is true. Enables recovery from failures.
    /// </remarks>
    public bool SaveIntermediateCheckpoints { get; set; } = true;

    /// <summary>
    /// Gets or sets the directory for intermediate checkpoints.
    /// </summary>
    /// <remarks>
    /// Default is "./checkpoints". Relative to working directory.
    /// </remarks>
    public string CheckpointDirectory { get; set; } = "./checkpoints";

    /// <summary>
    /// Gets or sets the maximum number of checkpoints to keep.
    /// </summary>
    /// <remarks>
    /// Default is 3. Older checkpoints are deleted to save disk space.
    /// </remarks>
    public int MaxCheckpointsToKeep { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to resume from the latest checkpoint.
    /// </summary>
    /// <remarks>
    /// Default is false. Set to true to continue interrupted training.
    /// </remarks>
    public bool ResumeFromCheckpoint { get; set; } = false;

    /// <summary>
    /// Gets or sets the specific checkpoint path to resume from.
    /// </summary>
    /// <remarks>
    /// Default is empty (use latest if ResumeFromCheckpoint is true).
    /// </remarks>
    public string ResumeCheckpointPath { get; set; } = string.Empty;

    // ========================================================================
    // Evaluation Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to run evaluation after each stage.
    /// </summary>
    /// <remarks>
    /// Default is true. Helps track progress between stages.
    /// </remarks>
    public bool EvaluateAfterEachStage { get; set; } = true;

    /// <summary>
    /// Gets or sets the evaluation metrics to track.
    /// </summary>
    /// <remarks>
    /// Default includes loss and perplexity.
    /// </remarks>
    public string[] EvaluationMetrics { get; set; } = new[] { "loss", "perplexity" };

    /// <summary>
    /// Gets or sets the evaluation data to use across all stages.
    /// </summary>
    /// <remarks>
    /// Default is null. Set this to use the same validation data across all stages.
    /// </remarks>
    public FineTuningData<T, TInput, TOutput>? GlobalEvaluationData { get; set; }

    // ========================================================================
    // Early Stopping Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the global early stopping configuration applied across stages.
    /// </summary>
    /// <remarks>
    /// Default configuration with patience=5 and monitoring loss.
    /// </remarks>
    public GlobalEarlyStoppingConfig GlobalEarlyStopping { get; set; } = new GlobalEarlyStoppingConfig();

    // ========================================================================
    // Callbacks
    // ========================================================================

    /// <summary>
    /// Gets or sets callback actions to execute before the pipeline starts.
    /// </summary>
    public Action<TrainingPipelineConfiguration<T, TInput, TOutput>>? OnPipelineStart { get; set; }

    /// <summary>
    /// Gets or sets callback actions to execute between stages.
    /// </summary>
    public List<Action<TrainingStageResult<T, TInput, TOutput>>>? InterStageCallbacks { get; set; }

    /// <summary>
    /// Gets or sets callback actions to execute when the pipeline completes.
    /// </summary>
    public Action<List<TrainingStageResult<T, TInput, TOutput>>>? OnPipelineComplete { get; set; }

    /// <summary>
    /// Gets or sets callback actions to execute on pipeline failure.
    /// </summary>
    public Action<Exception, TrainingStageResult<T, TInput, TOutput>?>? OnPipelineError { get; set; }

    // ========================================================================
    // Distributed Training Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets the distributed training strategy.
    /// </summary>
    /// <remarks>
    /// Default is None (single device). Use DDP for multi-GPU training.
    /// </remarks>
    public DistributedStrategy DistributedStrategy { get; set; } = DistributedStrategy.None;

    /// <summary>
    /// Gets or sets the number of devices/GPUs to use.
    /// </summary>
    /// <remarks>
    /// Default is 1. Set higher for distributed training.
    /// </remarks>
    public int NumDevices { get; set; } = 1;

    // ========================================================================
    // Logging Configuration
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to enable verbose logging.
    /// </summary>
    /// <remarks>
    /// Default is false. Enable for detailed debug output.
    /// </remarks>
    public bool VerboseLogging { get; set; } = false;

    /// <summary>
    /// Gets or sets the logging directory.
    /// </summary>
    /// <remarks>
    /// Default is "./logs". Relative to working directory.
    /// </remarks>
    public string LogDirectory { get; set; } = "./logs";

    /// <summary>
    /// Gets or sets whether to log to WandB or similar experiment trackers.
    /// </summary>
    /// <remarks>
    /// Default is false. Enable to use experiment tracking services.
    /// </remarks>
    public bool EnableExperimentTracking { get; set; } = false;

    /// <summary>
    /// Gets or sets the experiment name for tracking.
    /// </summary>
    /// <remarks>
    /// Default is empty (auto-generated from pipeline name).
    /// </remarks>
    public string ExperimentName { get; set; } = string.Empty;

    // ========================================================================
    // Metadata
    // ========================================================================

    /// <summary>
    /// Gets or sets custom metadata for the pipeline.
    /// </summary>
    /// <remarks>
    /// Empty by default. Use to store custom key-value pairs.
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Gets or sets tags for categorizing the pipeline.
    /// </summary>
    /// <remarks>
    /// Empty by default. Useful for filtering and organizing pipelines.
    /// </remarks>
    public string[] Tags { get; set; } = Array.Empty<string>();

    // ========================================================================
    // Basic Stage Addition Methods
    // ========================================================================

    /// <summary>
    /// Adds a training stage to the pipeline.
    /// </summary>
    /// <param name="stage">The stage to add.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddStage(TrainingStage<T, TInput, TOutput> stage)
    {
        Stages.Add(stage ?? throw new ArgumentNullException(nameof(stage)));
        return this;
    }

    /// <summary>
    /// Adds a supervised fine-tuning (SFT) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSFTStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Supervised Fine-Tuning",
            Description = "Train on labeled input-output pairs",
            StageType = TrainingStageType.SupervisedFineTuning,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds an instruction tuning stage (specialized SFT).
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddInstructionTuningStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Instruction Tuning",
            Description = "Fine-tune on instruction-following datasets",
            StageType = TrainingStageType.InstructionTuning,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Preference Optimization Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a Direct Preference Optimization (DPO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddDPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Direct Preference Optimization",
            Description = "Learn from preference pairs without a reward model",
            StageType = TrainingStageType.PreferenceOptimization,
            FineTuningMethod = FineTuningMethodType.DPO,
            DPOBeta = 0.1,
            UseReferenceModel = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds an Identity Preference Optimization (IPO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddIPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Identity Preference Optimization",
            Description = "DPO variant that addresses overfitting issues",
            StageType = TrainingStageType.PreferenceOptimization,
            FineTuningMethod = FineTuningMethodType.IPO,
            UseReferenceModel = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a Kahneman-Tversky Optimization (KTO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddKTOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Kahneman-Tversky Optimization",
            Description = "Works with unpaired preference data using prospect theory",
            StageType = TrainingStageType.PreferenceOptimization,
            FineTuningMethod = FineTuningMethodType.KTO,
            UseReferenceModel = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a Simple Preference Optimization (SimPO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSimPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Simple Preference Optimization",
            Description = "Reference-free preference optimization",
            StageType = TrainingStageType.PreferenceOptimization,
            FineTuningMethod = FineTuningMethodType.SimPO,
            UseReferenceModel = false // SimPO is reference-free
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a Contrastive Preference Optimization (CPO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddCPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Contrastive Preference Optimization",
            Description = "Uses contrastive learning for preference optimization",
            StageType = TrainingStageType.ContrastivePreference,
            FineTuningMethod = FineTuningMethodType.CPO
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a Robust DPO (R-DPO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRobustDPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Robust DPO",
            Description = "DPO with robustness to noisy preferences",
            StageType = TrainingStageType.PreferenceOptimization,
            FineTuningMethod = FineTuningMethodType.RDPO,
            UseReferenceModel = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds an Odds Ratio Preference Optimization (ORPO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddORPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Odds Ratio Preference Optimization",
            Description = "Combines SFT and preference learning in one stage",
            StageType = TrainingStageType.OddsRatioPreference,
            FineTuningMethod = FineTuningMethodType.ORPO,
            UseReferenceModel = false // ORPO is reference-free
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a generic preference optimization stage with configurable method.
    /// </summary>
    /// <param name="method">The preference optimization method to use.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddPreferenceStage(
        FineTuningMethodType method = FineTuningMethodType.DPO,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = $"Preference Optimization ({method})",
            StageType = TrainingStageType.PreferenceOptimization,
            FineTuningMethod = method,
            UseReferenceModel = IsReferenceModelRequired(method)
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Reinforcement Learning Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds an RLHF (PPO-based) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRLHFStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Reinforcement Learning from Human Feedback",
            Description = "PPO-based optimization with a reward model",
            StageType = TrainingStageType.ReinforcementLearning,
            FineTuningMethod = FineTuningMethodType.RLHF,
            PPOClipRange = 0.2,
            KLPenaltyCoefficient = 0.01,
            GAELambda = 0.95
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a PPO stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddPPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Proximal Policy Optimization",
            Description = "Policy gradient optimization with clipping",
            StageType = TrainingStageType.ProximalPolicyOptimization,
            FineTuningMethod = FineTuningMethodType.PPO,
            PPOClipRange = 0.2,
            PPOEpochsPerBatch = 4,
            GAELambda = 0.95
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a GRPO (Group Relative Policy Optimization) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddGRPOStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Group Relative Policy Optimization",
            Description = "DeepSeek's approach - uses relative rankings within groups",
            StageType = TrainingStageType.GroupRelativePolicyOptimization,
            FineTuningMethod = FineTuningMethodType.GRPO,
            GRPOGroupSize = 4,
            GRPOUseRelativeRewards = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds an RLAIF (RL from AI Feedback) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRLAIFStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Reinforcement Learning from AI Feedback",
            Description = "Uses AI-generated feedback instead of human feedback",
            StageType = TrainingStageType.ReinforcementLearningAIFeedback,
            FineTuningMethod = FineTuningMethodType.RLAIF
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a generic reinforcement learning stage.
    /// </summary>
    /// <param name="method">The RL method to use.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRLStage(
        FineTuningMethodType method = FineTuningMethodType.RLHF,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = $"Reinforcement Learning ({method})",
            StageType = TrainingStageType.ReinforcementLearning,
            FineTuningMethod = method
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Ranking and Sampling Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a Rejection Sampling Optimization (RSO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRejectionSamplingStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Rejection Sampling Optimization",
            Description = "Generate N responses, filter best with reward model, fine-tune",
            StageType = TrainingStageType.RejectionSampling,
            FineTuningMethod = FineTuningMethodType.RSO,
            RejectionSamplingN = 10,
            RejectionSamplingTopK = 1
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a Rank Responses to align Human Feedback (RRHF) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRRHFStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Rank Responses to align Human Feedback",
            Description = "Uses response rankings instead of pairwise preferences",
            StageType = TrainingStageType.RankResponses,
            FineTuningMethod = FineTuningMethodType.RRHF
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a Preference Ranking Optimization (PRO) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddPROStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Preference Ranking Optimization",
            Description = "Optimizes based on preference rankings over multiple responses",
            StageType = TrainingStageType.PreferenceRanking,
            FineTuningMethod = FineTuningMethodType.PRO
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a Sequence Likelihood Calibration (SLiC-HF) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSLiCStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Sequence Likelihood Calibration",
            Description = "Calibrates sequence likelihoods using human feedback",
            StageType = TrainingStageType.SequenceLikelihoodCalibration,
            FineTuningMethod = FineTuningMethodType.SLiCHF
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Constitutional and Safety Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a Constitutional AI stage.
    /// </summary>
    /// <param name="principles">The constitutional principles to use.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddConstitutionalAIStage(
        string[]? principles = null,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Constitutional AI",
            Description = "Self-improvement using constitutional principles",
            StageType = TrainingStageType.Constitutional,
            FineTuningMethod = FineTuningMethodType.ConstitutionalAI,
            ConstitutionalPrinciples = principles ?? GetDefaultConstitutionalPrinciples(),
            CritiqueRevisionRounds = 2
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a safety alignment stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSafetyAlignmentStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Safety Alignment",
            Description = "Focus on reducing harmful outputs",
            StageType = TrainingStageType.SafetyAlignment,
            FineTuningMethod = FineTuningMethodType.DPO
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a harmlessness training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddHarmlessnessStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Harmlessness Training",
            Description = "Train to refuse harmful requests appropriately",
            StageType = TrainingStageType.HarmlessnessTraining,
            FineTuningMethod = FineTuningMethodType.DPO
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a helpfulness training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddHelpfulnessStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Helpfulness Training",
            Description = "Maximize helpfulness within safety constraints",
            StageType = TrainingStageType.HelpfulnessTraining,
            FineTuningMethod = FineTuningMethodType.DPO
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Self-Play and Self-Improvement Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a Self-Play Fine-Tuning (SPIN) stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSPINStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Self-Play Fine-Tuning",
            Description = "Model learns to prefer human responses over its own",
            StageType = TrainingStageType.SelfPlay,
            FineTuningMethod = FineTuningMethodType.SPIN,
            SelfPlayIterations = 3,
            SelfPlayTemperature = 0.7,
            SelfPlayResponsesPerPrompt = 4
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a self-rewarding stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSelfRewardingStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Self-Rewarding Language Model",
            Description = "Model acts as both generator and reward model",
            StageType = TrainingStageType.SelfRewarding,
            FineTuningMethod = FineTuningMethodType.DPO
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Knowledge Distillation Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a knowledge distillation stage.
    /// </summary>
    /// <param name="teacherModel">The teacher model to distill from.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddDistillationStage(
        IFullModel<T, TInput, TOutput>? teacherModel = null,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Knowledge Distillation",
            Description = "Transfer knowledge from teacher to student model",
            StageType = TrainingStageType.KnowledgeDistillation,
            FineTuningMethod = FineTuningMethodType.KnowledgeDistillation,
            TeacherModel = teacherModel,
            DistillationTemperature = 2.0,
            DistillationAlpha = 0.5
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a synthetic data training stage.
    /// </summary>
    /// <param name="teacherModel">The teacher model to generate synthetic data.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddSyntheticDataStage(
        IFullModel<T, TInput, TOutput>? teacherModel = null,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Synthetic Data Training",
            Description = "Train on data generated by a teacher model",
            StageType = TrainingStageType.SyntheticDataTraining,
            FineTuningMethod = FineTuningMethodType.SFT,
            TeacherModel = teacherModel
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // LoRA and PEFT Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a LoRA adapter training stage.
    /// </summary>
    /// <param name="rank">LoRA rank (dimension of low-rank matrices).</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddLoRAStage(
        int rank = 16,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "LoRA Adapter Training",
            Description = "Parameter-efficient fine-tuning with low-rank adapters",
            StageType = TrainingStageType.LoRAAdapterTraining,
            FineTuningMethod = FineTuningMethodType.SFT,
            UseLoRA = true,
            LoRARank = rank,
            LoRAAlpha = rank * 2.0,
            LoRADropout = 0.05,
            FreezeBaseModel = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a QLoRA (quantized LoRA) stage.
    /// </summary>
    /// <param name="rank">LoRA rank.</param>
    /// <param name="bits">Quantization bits (4 or 8).</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddQLoRAStage(
        int rank = 16,
        int bits = 4,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "QLoRA Training",
            Description = "Quantized LoRA for memory-efficient training",
            StageType = TrainingStageType.LoRAAdapterTraining,
            FineTuningMethod = FineTuningMethodType.SFT,
            UseLoRA = true,
            UseQLoRA = true,
            LoRARank = rank,
            QLoRABits = bits,
            LoRAAlpha = rank * 2.0,
            FreezeBaseModel = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds an adapter merging stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddAdapterMergingStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Adapter Merging",
            Description = "Merge trained LoRA adapters into the base model",
            StageType = TrainingStageType.AdapterMerging,
            MergeLoRAAfterTraining = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Specialized Training Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a code fine-tuning stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddCodeFineTuningStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Code Fine-Tuning",
            Description = "Fine-tune on code with execution feedback",
            StageType = TrainingStageType.CodeFineTuning,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a math reasoning fine-tuning stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddMathReasoningStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Math Reasoning Fine-Tuning",
            Description = "Fine-tune on mathematical reasoning with verification",
            StageType = TrainingStageType.MathReasoningFineTuning,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a chain-of-thought training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddChainOfThoughtStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Chain-of-Thought Training",
            Description = "Train to produce step-by-step reasoning",
            StageType = TrainingStageType.ChainOfThoughtTraining,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a tool use training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddToolUseStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Tool Use Training",
            Description = "Train to use external tools",
            StageType = TrainingStageType.ToolUseTraining,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds an agentic behavior training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddAgenticStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Agentic Training",
            Description = "Train for multi-step autonomous task completion",
            StageType = TrainingStageType.AgenticTraining,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a multi-turn conversation training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddMultiTurnConversationStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Multi-Turn Conversation Training",
            Description = "Train on multi-turn dialogues for coherent conversations",
            StageType = TrainingStageType.MultiTurnConversation,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Reward Model Training Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds a reward model training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddRewardModelStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Reward Model Training",
            Description = "Train a model to predict human preferences",
            StageType = TrainingStageType.RewardModelTraining,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a process reward model (PRM) training stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddProcessRewardModelStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Process Reward Model Training",
            Description = "Train to evaluate reasoning steps, not just final outputs",
            StageType = TrainingStageType.ProcessRewardModelTraining,
            FineTuningMethod = FineTuningMethodType.SFT
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Utility Stage Methods
    // ========================================================================

    /// <summary>
    /// Adds an evaluation-only stage (no training, just metrics).
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddEvaluationStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Evaluation",
            Description = "Run evaluation metrics without training",
            StageType = TrainingStageType.Evaluation,
            IsEvaluationOnly = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a checkpoint stage.
    /// </summary>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddCheckpointStage(
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = "Checkpoint",
            Description = "Save checkpoint and run validation",
            StageType = TrainingStageType.Checkpoint,
            IsEvaluationOnly = true,
            SaveCheckpointAfter = true
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    /// <summary>
    /// Adds a custom training stage with user-defined logic.
    /// </summary>
    /// <param name="name">Name of the custom stage.</param>
    /// <param name="trainFunc">The custom training function.</param>
    /// <param name="configure">Optional configuration action for the stage.</param>
    /// <returns>This configuration for method chaining.</returns>
    public TrainingPipelineConfiguration<T, TInput, TOutput> AddCustomStage(
        string name,
        Func<IFullModel<T, TInput, TOutput>, FineTuningData<T, TInput, TOutput>, CancellationToken, Task<IFullModel<T, TInput, TOutput>>> trainFunc,
        Action<TrainingStage<T, TInput, TOutput>>? configure = null)
    {
        var stage = new TrainingStage<T, TInput, TOutput>
        {
            Name = name,
            StageType = TrainingStageType.Custom,
            CustomTrainingFunction = trainFunc
        };
        configure?.Invoke(stage);
        return AddStage(stage);
    }

    // ========================================================================
    // Industry-Standard Pipeline Factory Methods
    // ========================================================================

    /// <summary>
    /// Creates an OpenAI InstructGPT-style pipeline (SFT → Reward Model → PPO).
    /// </summary>
    /// <remarks>
    /// The original ChatGPT training pipeline from the InstructGPT paper.
    /// </remarks>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> InstructGPT(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null,
        FineTuningData<T, TInput, TOutput>? rlData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "InstructGPT Pipeline",
            Description = "OpenAI's InstructGPT training pipeline: SFT → Reward Model → PPO",
            EnableAutoSelection = false
        };

        // Stage 1: Supervised Fine-Tuning
        pipeline.AddSFTStage(stage =>
        {
            stage.Name = "InstructGPT SFT";
            stage.Description = "Train on human-written demonstrations";
            stage.TrainingData = sftData;
            stage.Epochs = 3;
            stage.BatchSize = 8;
        });

        // Stage 2: Reward Model Training
        pipeline.AddRewardModelStage(stage =>
        {
            stage.Name = "InstructGPT Reward Model";
            stage.Description = "Train reward model on human comparisons";
            stage.TrainingData = preferenceData;
            stage.Epochs = 1;
            stage.BatchSize = 64;
        });

        // Stage 3: PPO
        pipeline.AddPPOStage(stage =>
        {
            stage.Name = "InstructGPT PPO";
            stage.Description = "Optimize policy using reward model";
            stage.TrainingData = rlData;
            stage.PPOClipRange = 0.2;
            stage.KLPenaltyCoefficient = 0.02;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a Meta Llama 2-style pipeline (SFT → Rejection Sampling → DPO).
    /// </summary>
    /// <remarks>
    /// Meta's approach for Llama 2 Chat models.
    /// </remarks>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> Llama2(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Llama 2 Pipeline",
            Description = "Meta's Llama 2 training pipeline: SFT → Rejection Sampling → DPO",
            EnableAutoSelection = false
        };

        // Stage 1: SFT
        pipeline.AddSFTStage(stage =>
        {
            stage.Name = "Llama 2 SFT";
            stage.TrainingData = sftData;
            stage.Epochs = 2;
            stage.BatchSize = 128;
        });

        // Stage 2: Rejection Sampling
        pipeline.AddRejectionSamplingStage(stage =>
        {
            stage.Name = "Llama 2 Rejection Sampling";
            stage.RejectionSamplingN = 10;
            stage.RejectionSamplingTopK = 1;
        });

        // Stage 3: DPO (multiple iterations)
        for (int i = 1; i <= 5; i++)
        {
            pipeline.AddDPOStage(stage =>
            {
                stage.Name = $"Llama 2 DPO Iteration {i}";
                stage.TrainingData = preferenceData;
                stage.DPOBeta = 0.1;
                stage.Epochs = 1;
            });
        }

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates an Anthropic Claude-style pipeline (SFT → Constitutional AI → RLHF).
    /// </summary>
    /// <remarks>
    /// Anthropic's Constitutional AI approach for Claude models.
    /// </remarks>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> AnthropicClaude(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        string[]? principles = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Anthropic Claude Pipeline",
            Description = "Anthropic's Constitutional AI pipeline: SFT → CAI → RLHF",
            EnableAutoSelection = false
        };

        // Stage 1: SFT with HHH examples
        pipeline.AddSFTStage(stage =>
        {
            stage.Name = "Claude SFT (HHH)";
            stage.Description = "Train on Helpful, Harmless, Honest examples";
            stage.TrainingData = sftData;
            stage.Epochs = 3;
        });

        // Stage 2: Constitutional AI - critique and revision
        pipeline.AddConstitutionalAIStage(principles, stage =>
        {
            stage.Name = "Claude Constitutional AI";
            stage.CritiqueRevisionRounds = 2;
        });

        // Stage 3: RLAIF with constitutional principles
        pipeline.AddRLAIFStage(stage =>
        {
            stage.Name = "Claude RLAIF";
            stage.Description = "RL from AI Feedback using constitutional principles";
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a DeepSeek-style pipeline (SFT → GRPO).
    /// </summary>
    /// <remarks>
    /// DeepSeek's efficient training approach using GRPO instead of PPO.
    /// </remarks>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> DeepSeek(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? grpoData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "DeepSeek Pipeline",
            Description = "DeepSeek's training pipeline: SFT → GRPO",
            EnableAutoSelection = false
        };

        // Stage 1: SFT
        pipeline.AddSFTStage(stage =>
        {
            stage.Name = "DeepSeek SFT";
            stage.TrainingData = sftData;
            stage.Epochs = 2;
        });

        // Stage 2: GRPO
        pipeline.AddGRPOStage(stage =>
        {
            stage.Name = "DeepSeek GRPO";
            stage.TrainingData = grpoData;
            stage.GRPOGroupSize = 4;
            stage.GRPOUseRelativeRewards = true;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a standard SFT → DPO pipeline (most common alignment workflow).
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> StandardAlignment(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Standard Alignment Pipeline",
            Description = "Standard SFT → DPO alignment workflow",
            EnableAutoSelection = false
        };

        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 3;
            stage.BatchSize = 8;
        });

        pipeline.AddDPOStage(stage =>
        {
            stage.TrainingData = preferenceData;
            stage.Epochs = 1;
            stage.BatchSize = 4;
            stage.DPOBeta = 0.1;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a reference-free alignment pipeline using ORPO.
    /// </summary>
    /// <remarks>
    /// ORPO combines SFT and preference learning, requiring no reference model.
    /// More memory-efficient than DPO.
    /// </remarks>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> ORPOAlignment(
        FineTuningData<T, TInput, TOutput>? data = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "ORPO Alignment Pipeline",
            Description = "Reference-free alignment using ORPO (combines SFT + preference)",
            EnableAutoSelection = false
        };

        pipeline.AddORPOStage(stage =>
        {
            stage.TrainingData = data;
            stage.Epochs = 3;
            stage.BatchSize = 4;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a SimPO alignment pipeline (reference-free, simple).
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> SimPOAlignment(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "SimPO Alignment Pipeline",
            Description = "Reference-free preference optimization with SimPO",
            EnableAutoSelection = false
        };

        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 3;
        });

        pipeline.AddSimPOStage(stage =>
        {
            stage.TrainingData = preferenceData;
            stage.Epochs = 1;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a full RLHF pipeline (SFT → Reward Model → PPO).
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> FullRLHF(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? rlData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Full RLHF Pipeline",
            Description = "Complete RLHF pipeline: SFT → Reward Model → PPO",
            EnableAutoSelection = false
        };

        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 3;
            stage.BatchSize = 8;
        });

        pipeline.AddRewardModelStage(stage =>
        {
            stage.Epochs = 1;
        });

        pipeline.AddRLHFStage(stage =>
        {
            stage.TrainingData = rlData;
            stage.PPOClipRange = 0.2;
            stage.KLPenaltyCoefficient = 0.01;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a Constitutional AI pipeline (SFT → CAI critique/revision → preference).
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> ConstitutionalAI(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        string[]? principles = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Constitutional AI Pipeline",
            Description = "Constitutional AI training: SFT → critique/revision → RLAIF",
            EnableAutoSelection = false
        };

        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 2;
            stage.BatchSize = 8;
        });

        pipeline.AddConstitutionalAIStage(principles, stage =>
        {
            stage.CritiqueRevisionRounds = 2;
        });

        pipeline.AddRLAIFStage();

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a curriculum learning pipeline with progressively harder stages.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> CurriculumLearning(
        params (string Name, FineTuningData<T, TInput, TOutput> Data)[] curriculumStages)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Curriculum Learning Pipeline",
            Description = "Progressive training from easy to hard examples",
            EnableAutoSelection = false
        };

        for (int i = 0; i < curriculumStages.Length; i++)
        {
            var (name, data) = curriculumStages[i];
            double learningRateMultiplier = 1.0 / (1 + i * 0.2);

            pipeline.AddSFTStage(stage =>
            {
                stage.Name = $"Curriculum Stage {i + 1}: {name}";
                stage.StageType = TrainingStageType.CurriculumLearning;
                stage.TrainingData = data;
                stage.Epochs = 2;
                stage.BatchSize = 8;
                stage.LearningRate = 2e-5 * learningRateMultiplier;
            });
        }

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates an iterative refinement pipeline that runs multiple DPO rounds.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> IterativeRefinement(
        int iterations,
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null)
    {
        if (iterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(iterations), "Must have at least 1 iteration.");
        }

        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Iterative Refinement Pipeline",
            Description = $"SFT followed by {iterations} rounds of DPO refinement",
            EnableAutoSelection = false
        };

        // Initial SFT
        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 3;
            stage.BatchSize = 8;
        });

        // Multiple DPO iterations
        for (int i = 0; i < iterations; i++)
        {
            int iteration = i + 1;
            double beta = 0.1 / (1 + i * 0.1);

            pipeline.AddDPOStage(stage =>
            {
                stage.Name = $"DPO Iteration {iteration}";
                stage.TrainingData = preferenceData;
                stage.DPOBeta = beta;
                stage.Epochs = 1;
                stage.BatchSize = 4;
            });

            // Add evaluation between iterations
            pipeline.AddEvaluationStage(stage =>
            {
                stage.Name = $"Evaluation after DPO {iteration}";
            });
        }

        return pipeline;
    }

    /// <summary>
    /// Creates an iterative SPIN pipeline.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> IterativeSPIN(
        int iterations = 3,
        FineTuningData<T, TInput, TOutput>? sftData = null)
    {
        if (iterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(iterations), "Must have at least 1 iteration.");
        }

        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Iterative SPIN Pipeline",
            Description = $"Self-Play Fine-Tuning with {iterations} iterations",
            EnableAutoSelection = false
        };

        // Initial SFT
        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 3;
        });

        // SPIN iterations
        for (int i = 0; i < iterations; i++)
        {
            int iteration = i + 1;

            pipeline.AddSPINStage(stage =>
            {
                stage.Name = $"SPIN Iteration {iteration}";
                stage.SelfPlayIterations = 1;
                stage.SelfPlayTemperature = 0.7;
                stage.SelfPlayResponsesPerPrompt = 4;
            });

            pipeline.AddEvaluationStage(stage =>
            {
                stage.Name = $"Evaluation after SPIN {iteration}";
            });
        }

        return pipeline;
    }

    /// <summary>
    /// Creates a memory-efficient LoRA fine-tuning pipeline.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> LoRAFineTuning(
        int rank = 16,
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? preferenceData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "LoRA Fine-Tuning Pipeline",
            Description = "Parameter-efficient fine-tuning using LoRA adapters",
            EnableAutoSelection = false
        };

        // LoRA SFT
        pipeline.AddLoRAStage(rank, stage =>
        {
            stage.Name = "LoRA SFT";
            stage.TrainingData = sftData;
            stage.Epochs = 3;
            stage.FineTuningMethod = FineTuningMethodType.SFT;
        });

        // LoRA DPO (if preference data provided)
        if (preferenceData != null)
        {
            pipeline.AddLoRAStage(rank, stage =>
            {
                stage.Name = "LoRA DPO";
                stage.TrainingData = preferenceData;
                stage.FineTuningMethod = FineTuningMethodType.DPO;
                stage.DPOBeta = 0.1;
                stage.Epochs = 1;
            });
        }

        // Merge adapters
        pipeline.AddAdapterMergingStage();

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a QLoRA fine-tuning pipeline for maximum memory efficiency.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> QLoRAFineTuning(
        int rank = 16,
        int bits = 4,
        FineTuningData<T, TInput, TOutput>? data = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "QLoRA Fine-Tuning Pipeline",
            Description = $"Memory-efficient {bits}-bit quantized LoRA fine-tuning",
            EnableAutoSelection = false
        };

        pipeline.AddQLoRAStage(rank, bits, stage =>
        {
            stage.TrainingData = data;
            stage.Epochs = 3;
        });

        pipeline.AddAdapterMergingStage();
        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a distillation pipeline (teacher → student).
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> KnowledgeDistillation(
        IFullModel<T, TInput, TOutput>? teacherModel = null,
        FineTuningData<T, TInput, TOutput>? data = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Knowledge Distillation Pipeline",
            Description = "Transfer knowledge from teacher to student model",
            EnableAutoSelection = false
        };

        pipeline.AddDistillationStage(teacherModel, stage =>
        {
            stage.TrainingData = data;
            stage.DistillationTemperature = 2.0;
            stage.DistillationAlpha = 0.5;
            stage.Epochs = 5;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a code model training pipeline.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> CodeModel(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? codeExecutionData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Code Model Pipeline",
            Description = "Training pipeline for code generation models",
            EnableAutoSelection = false
        };

        // Instruction tuning on code
        pipeline.AddInstructionTuningStage(stage =>
        {
            stage.Name = "Code Instruction Tuning";
            stage.TrainingData = sftData;
            stage.Epochs = 3;
        });

        // Code-specific fine-tuning with execution feedback
        pipeline.AddCodeFineTuningStage(stage =>
        {
            stage.Name = "Code Execution Fine-Tuning";
            stage.TrainingData = codeExecutionData;
            stage.Epochs = 2;
        });

        // DPO on code preferences
        pipeline.AddDPOStage(stage =>
        {
            stage.Name = "Code DPO";
            stage.Epochs = 1;
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a math reasoning model pipeline.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> MathReasoning(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? cotData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Math Reasoning Pipeline",
            Description = "Training pipeline for mathematical reasoning",
            EnableAutoSelection = false
        };

        // SFT on math problems
        pipeline.AddSFTStage(stage =>
        {
            stage.Name = "Math SFT";
            stage.TrainingData = sftData;
            stage.Epochs = 3;
        });

        // Chain-of-thought training
        pipeline.AddChainOfThoughtStage(stage =>
        {
            stage.Name = "Math Chain-of-Thought";
            stage.TrainingData = cotData;
            stage.Epochs = 2;
        });

        // Math reasoning fine-tuning with verification
        pipeline.AddMathReasoningStage(stage =>
        {
            stage.Name = "Math Verification Fine-Tuning";
            stage.Epochs = 2;
        });

        // Process reward model for step-by-step verification
        pipeline.AddProcessRewardModelStage(stage =>
        {
            stage.Name = "Math Process Reward Model";
        });

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates a safety-focused training pipeline.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> SafetyFocused(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? safetyData = null,
        string[]? principles = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Safety-Focused Pipeline",
            Description = "Emphasis on safety and harm reduction",
            EnableAutoSelection = false
        };

        // Base SFT
        pipeline.AddSFTStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 2;
        });

        // Constitutional AI for safety
        pipeline.AddConstitutionalAIStage(principles ?? GetSafetyPrinciples());

        // Harmlessness training
        pipeline.AddHarmlessnessStage(stage =>
        {
            stage.TrainingData = safetyData;
        });

        // Helpfulness training (balanced with safety)
        pipeline.AddHelpfulnessStage();

        // Safety alignment
        pipeline.AddSafetyAlignmentStage();

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Creates an agent/tool-use training pipeline.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> AgentTraining(
        FineTuningData<T, TInput, TOutput>? sftData = null,
        FineTuningData<T, TInput, TOutput>? toolData = null,
        FineTuningData<T, TInput, TOutput>? agenticData = null)
    {
        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Agent Training Pipeline",
            Description = "Training pipeline for agentic AI with tool use",
            EnableAutoSelection = false
        };

        // Base instruction tuning
        pipeline.AddInstructionTuningStage(stage =>
        {
            stage.TrainingData = sftData;
            stage.Epochs = 2;
        });

        // Tool use training
        pipeline.AddToolUseStage(stage =>
        {
            stage.TrainingData = toolData;
            stage.Epochs = 2;
        });

        // Agentic behavior training
        pipeline.AddAgenticStage(stage =>
        {
            stage.TrainingData = agenticData;
            stage.Epochs = 2;
        });

        // Multi-turn conversation
        pipeline.AddMultiTurnConversationStage();

        pipeline.AddEvaluationStage();

        return pipeline;
    }

    /// <summary>
    /// Automatically selects an appropriate pipeline based on available data.
    /// </summary>
    public static TrainingPipelineConfiguration<T, TInput, TOutput> Auto(
        FineTuningData<T, TInput, TOutput> availableData)
    {
        if (availableData == null)
        {
            throw new ArgumentNullException(nameof(availableData));
        }

        var pipeline = new TrainingPipelineConfiguration<T, TInput, TOutput>
        {
            Name = "Auto-Selected Pipeline",
            Description = "Automatically configured based on available data",
            EnableAutoSelection = true
        };

        // Stage 1: Always start with SFT if we have supervised data
        if (availableData.HasSFTData)
        {
            pipeline.AddSFTStage(stage =>
            {
                stage.TrainingData = availableData;
                stage.Epochs = 3;
                stage.BatchSize = 8;
            });
        }

        // Stage 2: Add preference optimization if we have preference data
        if (availableData.HasPairwisePreferenceData)
        {
            pipeline.AddDPOStage(stage =>
            {
                stage.TrainingData = availableData;
                stage.Epochs = 1;
                stage.BatchSize = 4;
            });
        }
        else if (availableData.HasUnpairedPreferenceData)
        {
            pipeline.AddKTOStage(stage =>
            {
                stage.TrainingData = availableData;
                stage.Epochs = 1;
                stage.BatchSize = 4;
            });
        }

        // Stage 3: Add RL if we have reward data
        if (availableData.HasRLData)
        {
            pipeline.AddGRPOStage(stage =>
            {
                stage.TrainingData = availableData;
            });
        }

        // Stage 4: Constitutional AI if we have critique data
        if (availableData.CritiqueRevisions.Length > 0)
        {
            pipeline.AddConstitutionalAIStage();
        }

        // Always end with evaluation
        pipeline.AddEvaluationStage();

        return pipeline;
    }

    // ========================================================================
    // Pipeline Validation Methods
    // ========================================================================

    /// <summary>
    /// Validates the pipeline configuration.
    /// </summary>
    /// <returns>A list of validation errors, empty if valid.</returns>
    public List<string> Validate()
    {
        var errors = new List<string>();

        if (Stages.Count == 0 && !EnableAutoSelection)
        {
            errors.Add("Pipeline has no stages defined and auto-selection is disabled.");
        }

        for (int i = 0; i < Stages.Count; i++)
        {
            var stage = Stages[i];
            var stageErrors = ValidateStage(stage, i);
            errors.AddRange(stageErrors);
        }

        // Check for incompatible stage sequences
        for (int i = 1; i < Stages.Count; i++)
        {
            var curr = Stages[i];

            // PPO requires a reward model to have been trained
            if (curr.StageType == TrainingStageType.ProximalPolicyOptimization &&
                curr.RewardModel == null &&
                !HasPreviousRewardModelStage(i))
            {
                errors.Add($"Stage {i + 1} ({curr.Name}): PPO requires a reward model. " +
                           "Either provide one or add a RewardModelTraining stage before PPO.");
            }
        }

        return errors;
    }

    /// <summary>
    /// Throws an exception if the pipeline is invalid.
    /// </summary>
    public void ValidateOrThrow()
    {
        var errors = Validate();
        if (errors.Count > 0)
        {
            throw new InvalidOperationException(
                $"Pipeline validation failed with {errors.Count} error(s):\n" +
                string.Join("\n", errors.Select((e, i) => $"  {i + 1}. {e}")));
        }
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    private static bool IsReferenceModelRequired(FineTuningMethodType method)
    {
        return method switch
        {
            FineTuningMethodType.SimPO => false,
            FineTuningMethodType.ORPO => false,
            FineTuningMethodType.SFT => false,
            FineTuningMethodType.GRPO => false,
            _ => true
        };
    }

    private static string[] GetDefaultConstitutionalPrinciples()
    {
        return new[]
        {
            "Please choose the response that is most helpful, accurate, and harmless.",
            "Please choose the response that is honest and does not make up information.",
            "Please choose the response that is less toxic, rude, or harmful.",
            "Please choose the response that respects user privacy and autonomy.",
            "Please choose the response that is more ethical and less biased."
        };
    }

    private static string[] GetSafetyPrinciples()
    {
        return new[]
        {
            "Choose the response that refuses to help with illegal activities.",
            "Choose the response that does not provide dangerous or harmful information.",
            "Choose the response that respects human dignity and rights.",
            "Choose the response that does not discriminate or promote hate.",
            "Choose the response that prioritizes user safety and well-being."
        };
    }

    private List<string> ValidateStage(TrainingStage<T, TInput, TOutput> stage, int index)
    {
        var errors = new List<string>();
        string prefix = $"Stage {index + 1} ({stage.Name}):";

        if (string.IsNullOrWhiteSpace(stage.Name))
        {
            errors.Add($"{prefix} Stage name is required.");
        }

        if (stage.Epochs <= 0)
        {
            errors.Add($"{prefix} Epochs must be positive.");
        }

        if (stage.BatchSize <= 0)
        {
            errors.Add($"{prefix} Batch size must be positive.");
        }

        if (stage.GradientAccumulationSteps <= 0)
        {
            errors.Add($"{prefix} Gradient accumulation steps must be positive.");
        }

        if (stage.UseLoRA)
        {
            if (stage.LoRARank <= 0)
            {
                errors.Add($"{prefix} LoRA rank must be positive.");
            }

            if (stage.UseQLoRA && stage.QLoRABits != 4 && stage.QLoRABits != 8)
            {
                errors.Add($"{prefix} QLoRA bits must be 4 or 8.");
            }
        }

        if (stage.StageType == TrainingStageType.Custom && stage.CustomTrainingFunction == null)
        {
            errors.Add($"{prefix} Custom stages require a CustomTrainingFunction.");
        }

        return errors;
    }

    private bool HasPreviousRewardModelStage(int currentIndex)
    {
        for (int i = 0; i < currentIndex; i++)
        {
            if (Stages[i].StageType == TrainingStageType.RewardModelTraining ||
                Stages[i].StageType == TrainingStageType.ProcessRewardModelTraining ||
                Stages[i].StageType == TrainingStageType.OutcomeRewardModelTraining)
            {
                return true;
            }
        }
        return false;
    }
}
