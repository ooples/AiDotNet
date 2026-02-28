using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for BayTransProto (Bayesian Transductive Prototypical Networks).
/// </summary>
/// <remarks>
/// BayTransProto extends prototypical networks with Bayesian parameter posteriors and
/// transductive refinement. The adapted parameters are sampled from a learned posterior,
/// and transductive steps use query predictions to iteratively refine the posterior mean.
/// </remarks>
public class BayTransProtoOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Number of posterior samples during training. Default: 5.</summary>
    public int NumPosteriorSamples { get; set; } = 5;

    /// <summary>Initial log-variance for posterior. Default: -3.0.</summary>
    public double InitialLogVar { get; set; } = -3.0;

    /// <summary>Transductive refinement steps using query predictions. Default: 3.</summary>
    public int TransductiveSteps { get; set; } = 3;

    /// <summary>Learning rate for transductive refinement. Default: 0.01.</summary>
    public double TransductiveLR { get; set; } = 0.01;

    /// <summary>KL divergence weight for posterior regularization. Default: 0.01.</summary>
    public double KLWeight { get; set; } = 0.01;

    public BayTransProtoOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new BayTransProtoOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumPosteriorSamples = NumPosteriorSamples, InitialLogVar = InitialLogVar,
        TransductiveSteps = TransductiveSteps, TransductiveLR = TransductiveLR, KLWeight = KLWeight
    };
}
