using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the MetaContinualAL (Meta-Continual Active Learning) algorithm.
/// </summary>
/// <remarks>
/// MetaContinualAL combines active learning sample selection with continual meta-learning.
/// It uses gradient-norm-based uncertainty estimation to identify the most informative
/// parameter dimensions, focusing adaptation on high-uncertainty regions while maintaining
/// a running calibration of uncertainty statistics.
/// </remarks>
public class MetaContinualALOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Weight on the uncertainty-guided gradient scaling. Higher values amplify
    /// the learning signal for uncertain parameter dimensions. Default: 1.0.
    /// </summary>
    public double UncertaintyWeight { get; set; } = 1.0;

    /// <summary>
    /// EMA decay for running uncertainty statistics (mean/variance of gradient norms).
    /// Values close to 1.0 give more stable calibration. Default: 0.95.
    /// </summary>
    public double UncertaintyDecay { get; set; } = 0.95;

    /// <summary>
    /// Fraction of parameters (by highest uncertainty) to focus adaptation on.
    /// Remaining parameters get reduced learning rates. Default: 0.5.
    /// </summary>
    public double AcquisitionFraction { get; set; } = 0.5;

    /// <summary>
    /// Exploration bonus added to uncertain dimensions to encourage coverage.
    /// Acts as additive noise proportional to uncertainty. Default: 0.1.
    /// </summary>
    public double ExplorationBonus { get; set; } = 0.1;

    public MetaContinualALOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new MetaContinualALOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        UncertaintyWeight = UncertaintyWeight, UncertaintyDecay = UncertaintyDecay,
        AcquisitionFraction = AcquisitionFraction, ExplorationBonus = ExplorationBonus
    };
}
