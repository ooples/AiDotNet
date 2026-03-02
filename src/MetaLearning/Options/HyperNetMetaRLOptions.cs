using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for HyperNet Meta-RL: hypernetwork-based policy generation
/// for meta-reinforcement learning.
/// </summary>
/// <remarks>
/// <para>
/// HyperNet Meta-RL uses a hypernetwork to generate task-specific policy parameters
/// from a task embedding. The task embedding is computed from initial task interaction
/// data (support set), and the hypernetwork transforms it into a full parameter
/// vector for the policy network, enabling single-forward-pass adaptation.
/// </para>
/// </remarks>
public class HyperNetMetaRLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Dimensionality of the task embedding. Default: 32.</summary>
    public int TaskEmbeddingDim { get; set; } = 32;

    /// <summary>Hidden dimension of the hypernetwork. Default: 64.</summary>
    public int HyperNetHiddenDim { get; set; } = 64;

    /// <summary>Weight for the parameter regularization loss. Default: 0.001.</summary>
    public double ParamRegWeight { get; set; } = 0.001;

    public HyperNetMetaRLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && TaskEmbeddingDim > 0 && HyperNetHiddenDim > 0;
    public IMetaLearnerOptions<T> Clone() => new HyperNetMetaRLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        TaskEmbeddingDim = TaskEmbeddingDim, HyperNetHiddenDim = HyperNetHiddenDim, ParamRegWeight = ParamRegWeight
    };
}
