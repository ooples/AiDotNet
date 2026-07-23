using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-FDMixup: Feature-Distribution Mixup for cross-domain
/// few-shot learning (Xu et al., CVPR 2021).
/// </summary>
/// <remarks>
/// <para>
/// Meta-FDMixup improves cross-domain generalization by mixing feature distributions
/// (gradient signals) between tasks in a meta-batch. Instead of mixing raw inputs,
/// it mixes the gradient directions from different tasks, encouraging the meta-learner
/// to find an initialization robust across diverse task distributions.
/// </para>
/// </remarks>
public class MetaFDMixupOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Alpha parameter for the Beta distribution used to sample mixup coefficients.
    /// Higher values → mixup ratios closer to 0.5. Default: 2.0.
    /// </summary>
    public double MixupAlpha { get; set; } = 2.0;

    /// <summary>
    /// Probability of applying mixup to each task's gradient during inner loop. Default: 0.5.
    /// </summary>
    public double MixupProbability { get; set; } = 0.5;

    /// <summary>
    /// Weight for the feature distribution alignment loss. Default: 0.1.
    /// </summary>
    public double AlignmentWeight { get; set; } = 0.1;

    public MetaFDMixupOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && MixupAlpha > 0 && MixupProbability >= 0 && MixupProbability <= 1;
    public IMetaLearnerOptions<T> Clone() => new MetaFDMixupOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        MixupAlpha = MixupAlpha, MixupProbability = MixupProbability, AlignmentWeight = AlignmentWeight
    };
}
