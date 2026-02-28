using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the MetaTask (Meta-learned Task Augmentation) algorithm.
/// </summary>
/// <remarks>
/// MetaTask generates synthetic tasks by interpolating gradients between pairs of real tasks
/// using Beta-distributed mixing coefficients. The synthetic tasks augment the task distribution,
/// improving generalization of the meta-learned initialization.
/// </remarks>
public class MetaTaskOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    public double InnerLearningRate { get; set; } = 0.01;
    public double OuterLearningRate { get; set; } = 0.001;
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
    /// Alpha parameter for Beta(α,α) distribution used to sample interpolation coefficients.
    /// Higher values concentrate mixing around λ=0.5; lower values allow extreme mixing.
    /// Default: 2.0.
    /// </summary>
    public double InterpolationAlpha { get; set; } = 2.0;

    /// <summary>
    /// Number of synthetic (interpolated) tasks generated per meta-batch.
    /// These are added to the real tasks in the meta-objective. Default: 2.
    /// </summary>
    public int NumSyntheticTasks { get; set; } = 2;

    /// <summary>
    /// Weight on the synthetic task losses relative to real task losses.
    /// L_meta = L_real + SyntheticWeight * L_synthetic. Default: 0.5.
    /// </summary>
    public double SyntheticWeight { get; set; } = 0.5;

    public MetaTaskOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new MetaTaskOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        InterpolationAlpha = InterpolationAlpha, NumSyntheticTasks = NumSyntheticTasks,
        SyntheticWeight = SyntheticWeight
    };
}
