using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the MOCA (Meta-learning with Online Complementary Augmentation) algorithm.
/// </summary>
/// <remarks>
/// MOCA augments tasks in gradient space using complementary perturbations derived from
/// historical gradient statistics. The augmented gradients explore directions orthogonal
/// to the original task gradient, encouraging more robust meta-learned initializations.
/// </remarks>
public class MOCAOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Magnitude of gradient-space perturbation for augmented tasks.
    /// Perturbation is scaled by AugmentationStrength * gradient_history_std. Default: 0.1.
    /// </summary>
    public double AugmentationStrength { get; set; } = 0.1;

    /// <summary>
    /// EMA momentum for accumulating gradient history statistics (mean and variance).
    /// Default: 0.9.
    /// </summary>
    public double HistoryMomentum { get; set; } = 0.9;

    /// <summary>
    /// Number of augmented task variants generated per real task in the meta-batch.
    /// Total tasks processed = original + NumAugmentedTasks. Default: 1.
    /// </summary>
    public int NumAugmentedTasks { get; set; } = 1;

    /// <summary>
    /// Weight on the augmented task loss in the meta-objective.
    /// L_meta = L_original + ComplementaryWeight * L_augmented. Default: 0.5.
    /// </summary>
    public double ComplementaryWeight { get; set; } = 0.5;

    public MOCAOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new MOCAOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        AugmentationStrength = AugmentationStrength, HistoryMomentum = HistoryMomentum,
        NumAugmentedTasks = NumAugmentedTasks, ComplementaryWeight = ComplementaryWeight
    };
}
