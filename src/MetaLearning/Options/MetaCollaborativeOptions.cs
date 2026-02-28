using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-Collaborative Learning across multiple task domains.
/// </summary>
/// <remarks>
/// <para>
/// Meta-Collaborative Learning uses gradient alignment between concurrently adapted tasks
/// to transfer cross-domain knowledge. Tasks with aligned gradient directions reinforce each
/// other, while conflicting gradients are dampened. A domain-specific momentum buffer per task
/// stabilizes cross-task transfer.
/// </para>
/// </remarks>
public class MetaCollaborativeOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Weight for the gradient alignment loss between tasks. Default: 0.5.
    /// </summary>
    public double AlignmentWeight { get; set; } = 0.5;

    /// <summary>
    /// Momentum coefficient for the domain-specific gradient buffer. Default: 0.9.
    /// </summary>
    public double GradientMomentum { get; set; } = 0.9;

    /// <summary>
    /// Number of domain slots for maintaining separate gradient histories. Default: 4.
    /// </summary>
    public int NumDomainSlots { get; set; } = 4;

    public MetaCollaborativeOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && NumDomainSlots > 0;
    public IMetaLearnerOptions<T> Clone() => new MetaCollaborativeOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        AlignmentWeight = AlignmentWeight, GradientMomentum = GradientMomentum, NumDomainSlots = NumDomainSlots
    };
}
