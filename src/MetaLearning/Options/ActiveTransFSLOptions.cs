using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for ActiveTransFSL (Active Transductive Few-Shot Learning).
/// </summary>
/// <remarks>
/// ActiveTransFSL combines active learning with transductive inference. After initial
/// adaptation on support data, it uses gradient-norm-based uncertainty to identify the most
/// uncertain parameter dimensions, then performs transductive refinement steps that focus
/// adaptation on these uncertain regions using query data feedback.
/// </remarks>
public class ActiveTransFSLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Fraction of parameters (by uncertainty) to actively refine. Default: 0.5.</summary>
    public double SelectionFraction { get; set; } = 0.5;

    /// <summary>Number of transductive refinement steps using query gradients. Default: 3.</summary>
    public int TransductiveRefinementSteps { get; set; } = 3;

    /// <summary>Weight on the transductive (query-feedback) loss. Default: 0.5.</summary>
    public double TransductiveWeight { get; set; } = 0.5;

    /// <summary>Learning rate for transductive refinement. Default: 0.005.</summary>
    public double TransductiveLR { get; set; } = 0.005;

    public ActiveTransFSLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new ActiveTransFSLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        SelectionFraction = SelectionFraction, TransductiveRefinementSteps = TransductiveRefinementSteps,
        TransductiveWeight = TransductiveWeight, TransductiveLR = TransductiveLR
    };
}
