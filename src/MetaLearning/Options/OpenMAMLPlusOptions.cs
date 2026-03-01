using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Open-MAML++: MAML extended for open-set recognition with
/// novelty detection.
/// </summary>
/// <remarks>
/// <para>
/// Open-MAML++ extends MAML to handle open-set scenarios where test tasks may contain
/// classes not seen during meta-training. It meta-learns per-parameter learning rates
/// (like MAML++) and a novelty detection threshold based on prediction entropy. During
/// adaptation, predictions with entropy above the threshold are flagged as novel/unknown.
/// </para>
/// </remarks>
public class OpenMAMLPlusOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Initial novelty threshold (entropy-based). Default: 1.0.
    /// </summary>
    public double InitialNoveltyThreshold { get; set; } = 1.0;

    /// <summary>
    /// Weight for the entropy regularization loss. Default: 0.1.
    /// </summary>
    public double EntropyRegWeight { get; set; } = 0.1;

    /// <summary>
    /// Whether to meta-learn per-parameter learning rates (MAML++ style). Default: true.
    /// </summary>
    public bool LearnPerParamLR { get; set; } = true;

    /// <summary>
    /// Multi-step loss coefficient: weight for intermediate adaptation step losses. Default: 0.5.
    /// </summary>
    public double MultiStepLossWeight { get; set; } = 0.5;

    public OpenMAMLPlusOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && InitialNoveltyThreshold > 0;
    public IMetaLearnerOptions<T> Clone() => new OpenMAMLPlusOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        InitialNoveltyThreshold = InitialNoveltyThreshold, EntropyRegWeight = EntropyRegWeight,
        LearnPerParamLR = LearnPerParamLR, MultiStepLossWeight = MultiStepLossWeight
    };
}
