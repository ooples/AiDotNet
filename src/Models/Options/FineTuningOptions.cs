namespace AiDotNet.Models.Options;

using AiDotNet.Interfaces;

/// <summary>
/// Configuration options for fine-tuning methods.
/// </summary>
/// <remarks>
/// <para>
/// This class provides a comprehensive set of options that cover all fine-tuning method categories.
/// Each method type uses the relevant subset of options.
/// </para>
/// <para><b>For Beginners:</b> These settings control how the fine-tuning process works.
/// Most settings have sensible defaults based on research papers, so you can start with
/// the defaults and adjust as needed.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class FineTuningOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the fine-tuning method type.
    /// </summary>
    public FineTuningMethodType MethodType { get; set; } = FineTuningMethodType.SFT;

    // ========== Common Training Parameters ==========

    /// <summary>
    /// Gets or sets the learning rate for fine-tuning.
    /// </summary>
    /// <remarks>
    /// Default: 1e-5 (suitable for most preference methods)
    /// </remarks>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    public int BatchSize { get; set; } = 8;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    public int Epochs { get; set; } = 3;

    /// <summary>
    /// Gets or sets the gradient accumulation steps.
    /// </summary>
    /// <remarks>
    /// Allows effective batch sizes larger than memory permits.
    /// </remarks>
    public int GradientAccumulationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum gradient norm for clipping.
    /// </summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the warmup ratio for learning rate scheduling.
    /// </summary>
    public double WarmupRatio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 0.01;

    // ========== DPO-Family Parameters ==========

    /// <summary>
    /// Gets or sets the beta parameter for DPO-family methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the strength of the KL penalty from the reference model.
    /// Higher beta = stay closer to reference model.
    /// </para>
    /// <para>Default: 0.1 (DPO paper recommendation)</para>
    /// </remarks>
    public double Beta { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the label smoothing factor for preference learning.
    /// </summary>
    /// <remarks>
    /// Default: 0.0 (no smoothing). Values like 0.1 can help with noisy preferences.
    /// </remarks>
    public double LabelSmoothing { get; set; } = 0.0;

    // ========== SimPO-Specific Parameters ==========

    /// <summary>
    /// Gets or sets the gamma parameter for SimPO length normalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SimPO uses average log probability instead of sum, and gamma controls
    /// the target reward margin.
    /// </para>
    /// <para>Default: 0.5 (SimPO paper recommendation)</para>
    /// </remarks>
    public double SimPOGamma { get; set; } = 0.5;

    // ========== ORPO-Specific Parameters ==========

    /// <summary>
    /// Gets or sets the lambda parameter for ORPO odds ratio loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the weight of the odds ratio loss relative to the SFT loss.
    /// </para>
    /// <para>Default: 0.1 (ORPO paper recommendation)</para>
    /// </remarks>
    public double ORPOLambda { get; set; } = 0.1;

    // ========== KTO-Specific Parameters ==========

    /// <summary>
    /// Gets or sets the desirable weight for KTO.
    /// </summary>
    /// <remarks>
    /// <para>
    /// KTO uses prospect theory with separate weights for desirable/undesirable.
    /// </para>
    /// <para>Default: 1.0</para>
    /// </remarks>
    public double KTODesirableWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the undesirable weight for KTO.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Typically higher to emphasize avoiding bad outputs (loss aversion).
    /// </para>
    /// <para>Default: 1.0</para>
    /// </remarks>
    public double KTOUndesirableWeight { get; set; } = 1.0;

    // ========== RL-Based Parameters (RLHF, PPO, GRPO) ==========

    /// <summary>
    /// Gets or sets the KL coefficient for RL-based methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the KL penalty to prevent the model from diverging too far from the reference.
    /// </para>
    /// <para>Default: 0.02</para>
    /// </remarks>
    public double KLCoefficient { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the PPO clip range.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Standard PPO clipping parameter for policy updates.
    /// </para>
    /// <para>Default: 0.2</para>
    /// </remarks>
    public double PPOClipRange { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the number of PPO epochs per batch.
    /// </summary>
    public int PPOEpochsPerBatch { get; set; } = 4;

    /// <summary>
    /// Gets or sets the GAE lambda for advantage estimation.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.95 (standard for PPO)</para>
    /// </remarks>
    public double GAELambda { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the discount factor for rewards.
    /// </summary>
    public double Gamma { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the value function coefficient for PPO.
    /// </summary>
    public double ValueCoefficient { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the entropy coefficient for exploration.
    /// </summary>
    public double EntropyCoefficient { get; set; } = 0.01;

    // ========== GRPO-Specific Parameters ==========

    /// <summary>
    /// Gets or sets the group size for GRPO sampling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Number of responses to generate per prompt for group comparison.
    /// </para>
    /// <para>Default: 8 (DeepSeek recommendation)</para>
    /// </remarks>
    public int GRPOGroupSize { get; set; } = 8;

    /// <summary>
    /// Gets or sets the GRPO temperature for sampling.
    /// </summary>
    public double GRPOTemperature { get; set; } = 0.7;

    // ========== Constitutional AI Parameters ==========

    /// <summary>
    /// Gets or sets the constitutional principles for CAI methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These principles guide the model's self-critique and revision process.
    /// </para>
    /// </remarks>
    public string[] ConstitutionalPrinciples { get; set; } = new[]
    {
        "Choose responses that are helpful and informative",
        "Avoid responses that could cause harm to individuals or groups",
        "Be honest and do not make up false information",
        "Respect privacy and do not reveal personal information",
        "Be respectful and do not use offensive language"
    };

    /// <summary>
    /// Gets or sets the number of critique-revision iterations.
    /// </summary>
    public int CritiqueIterations { get; set; } = 2;

    // ========== Ranking-Based Parameters (RSO, RRHF) ==========

    /// <summary>
    /// Gets or sets the margin for ranking loss.
    /// </summary>
    public double RankingMargin { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the temperature for ranking softmax.
    /// </summary>
    public double RankingTemperature { get; set; } = 1.0;

    // ========== Self-Play Parameters (SPIN) ==========

    /// <summary>
    /// Gets or sets the number of self-play iterations.
    /// </summary>
    public int SPINIterations { get; set; } = 3;

    // ========== Knowledge Distillation Parameters ==========

    /// <summary>
    /// Gets or sets the distillation temperature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Higher temperature produces softer probability distributions.
    /// </para>
    /// <para>Default: 2.0</para>
    /// </remarks>
    public double DistillationTemperature { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the alpha weight between hard and soft labels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Alpha = 1.0 uses only soft labels, Alpha = 0.0 uses only hard labels.
    /// </para>
    /// <para>Default: 0.5 (balanced)</para>
    /// </remarks>
    public double DistillationAlpha { get; set; } = 0.5;

    // ========== LoRA/PEFT Integration ==========

    /// <summary>
    /// Gets or sets whether to use LoRA for parameter-efficient fine-tuning.
    /// </summary>
    public bool UseLoRA { get; set; } = false;

    /// <summary>
    /// Gets or sets the LoRA configuration when UseLoRA is true.
    /// </summary>
    public LoRAConfiguration? LoRAConfig { get; set; }

    // ========== Logging and Checkpointing ==========

    /// <summary>
    /// Gets or sets the logging frequency in steps.
    /// </summary>
    public int LoggingSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the checkpoint frequency in steps.
    /// </summary>
    public int CheckpointSteps { get; set; } = 500;

    /// <summary>
    /// Gets or sets the maximum number of checkpoints to keep.
    /// </summary>
    public int MaxCheckpoints { get; set; } = 3;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; }

    // ========== Advanced Parameters ==========

    /// <summary>
    /// Gets or sets the maximum sequence length.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets whether to use mixed precision training.
    /// </summary>
    public bool UseMixedPrecision { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to compile the model for faster training.
    /// </summary>
    public bool CompileModel { get; set; } = false;
}

/// <summary>
/// LoRA configuration for parameter-efficient fine-tuning.
/// </summary>
public class LoRAConfiguration
{
    /// <summary>
    /// Gets or sets the LoRA rank (r).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Lower rank = fewer parameters but less expressiveness.
    /// </para>
    /// <para>Default: 16 (good balance)</para>
    /// </remarks>
    public int Rank { get; set; } = 16;

    /// <summary>
    /// Gets or sets the LoRA alpha scaling factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: 32 (alpha/rank = 2 is common)</para>
    /// </remarks>
    public int Alpha { get; set; } = 32;

    /// <summary>
    /// Gets or sets the LoRA dropout rate.
    /// </summary>
    public double Dropout { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets which modules to apply LoRA to.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common choices: ["q_proj", "v_proj"] for attention-only,
    /// or ["q_proj", "k_proj", "v_proj", "o_proj"] for full attention.
    /// </para>
    /// </remarks>
    public string[] TargetModules { get; set; } = new[] { "q_proj", "v_proj" };

    /// <summary>
    /// Gets or sets whether to use bias terms in LoRA layers.
    /// </summary>
    public string BiasMode { get; set; } = "none";

    /// <summary>
    /// Gets or sets whether to use quantization (QLoRA).
    /// </summary>
    public bool UseQuantization { get; set; } = false;

    /// <summary>
    /// Gets or sets the quantization bits when UseQuantization is true.
    /// </summary>
    public int QuantizationBits { get; set; } = 4;
}
