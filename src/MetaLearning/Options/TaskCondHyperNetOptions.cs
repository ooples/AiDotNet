using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>Configuration options for Task-Conditioned HyperNetwork.</summary>
/// <remarks>
/// <para>
/// A hypernetwork generates task-specific parameter deltas conditioned on a task embedding
/// derived from support-set gradient statistics. The hypernetwork is a 2-layer MLP:
/// embedding → hidden → parameter delta, added to the meta-initialization.
/// </para>
/// </remarks>
public class TaskCondHyperNetOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Dimension of the task embedding derived from gradient statistics.
    /// Default: 32.
    /// </summary>
    public int EmbeddingDim { get; set; } = 32;

    /// <summary>
    /// Hidden dimension of the hypernetwork MLP.
    /// Default: 64.
    /// </summary>
    public int HyperHiddenDim { get; set; } = 64;

    /// <summary>
    /// Chunk size for chunked hypernetwork output (generates parameters in chunks
    /// to reduce hypernetwork size). Each chunk head maps hidden → chunkSize.
    /// Default: 16.
    /// </summary>
    public int ChunkSize { get; set; } = 16;

    /// <summary>
    /// L2 regularization weight on task embedding magnitude.
    /// Default: 0.01.
    /// </summary>
    public double EmbeddingRegWeight { get; set; } = 0.01;

    public TaskCondHyperNetOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
                              && EmbeddingDim > 0 && HyperHiddenDim > 0 && ChunkSize > 0;
    public IMetaLearnerOptions<T> Clone() => new TaskCondHyperNetOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        EmbeddingDim = EmbeddingDim, HyperHiddenDim = HyperHiddenDim,
        ChunkSize = ChunkSize, EmbeddingRegWeight = EmbeddingRegWeight
    };
}
