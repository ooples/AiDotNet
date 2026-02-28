using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the MePo (Memory Prototypes) meta-learning algorithm.
/// </summary>
/// <remarks>
/// MePo maintains a memory bank of gradient-space prototypes from previously seen tasks.
/// When adapting to a new task, it retrieves the nearest prototypes and uses them to
/// warm-start adaptation and regularize the inner loop toward known good trajectories.
/// </remarks>
public class MePoOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Maximum number of prototypes stored in the memory bank.
    /// Oldest prototypes are replaced when memory is full. Default: 32.
    /// </summary>
    public int MemorySize { get; set; } = 32;

    /// <summary>
    /// Dimensionality of each prototype vector (compressed gradient space).
    /// Default: 32.
    /// </summary>
    public int PrototypeDim { get; set; } = 32;

    /// <summary>
    /// Regularization weight pulling adapted parameters toward the trajectory
    /// suggested by the nearest retrieved prototypes. Default: 0.1.
    /// </summary>
    public double PrototypeRegWeight { get; set; } = 0.1;

    /// <summary>
    /// Number of nearest prototypes to retrieve for warm-starting and regularization.
    /// Default: 3.
    /// </summary>
    public int RetrievalTopK { get; set; } = 3;

    public MePoOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new MePoOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        MemorySize = MemorySize, PrototypeDim = PrototypeDim,
        PrototypeRegWeight = PrototypeRegWeight, RetrievalTopK = RetrievalTopK
    };
}
