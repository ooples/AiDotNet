using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot
/// Learning (Zhang et al., AAAI 2024).
/// </summary>
/// <remarks>
/// <para>
/// MetaDiff reframes the inner-loop gradient descent of meta-learning as a reverse diffusion
/// process over model weights. A task-conditional denoising network iteratively removes noise
/// to produce task-specific weights, conditioned on support-set features.
/// </para>
/// </remarks>
public class MetaDiffOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    public double InnerLearningRate { get; set; } = 0.01;
    public double OuterLearningRate { get; set; } = 0.0001;
    public int AdaptationSteps { get; set; } = 5;
    public int MetaBatchSize { get; set; } = 4;
    public int NumMetaIterations { get; set; } = 1000;
    public double? GradientClipThreshold { get; set; } = 10.0;
    public int? RandomSeed { get => Seed; set => Seed = value; }
    public int EvaluationTasks { get; set; } = 100;
    public int EvaluationFrequency { get; set; } = 100;
    public bool EnableCheckpointing { get; set; } = false;
    public int CheckpointFrequency { get; set; } = 500;
    public bool UseFirstOrder { get; set; } = true;
    public ILossFunction<T>? LossFunction { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Number of diffusion timesteps for the full forward/reverse process. Default: 100.
    /// During inference, fewer steps can be used (controlled by <see cref="SamplingSteps"/>).
    /// </summary>
    public int DiffusionSteps { get; set; } = 100;

    /// <summary>
    /// Number of denoising steps used during inference (≤ DiffusionSteps). Default: 20.
    /// </summary>
    public int SamplingSteps { get; set; } = 20;

    /// <summary>
    /// Starting value of the linear noise schedule β_1. Default: 0.0001.
    /// </summary>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Ending value of the linear noise schedule β_T. Default: 0.02.
    /// </summary>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Dimensionality of the task conditioning vector computed from support features. Default: 64.
    /// </summary>
    public int TaskConditionDim { get; set; } = 64;

    public MetaDiffOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && DiffusionSteps > 0 && SamplingSteps > 0;
    public IMetaLearnerOptions<T> Clone() => new MetaDiffOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        DiffusionSteps = DiffusionSteps, SamplingSteps = SamplingSteps, BetaStart = BetaStart,
        BetaEnd = BetaEnd, TaskConditionDim = TaskConditionDim
    };
}
