using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Unsupervised Meta-Learning (Hsu et al., 2019).
/// </summary>
/// <remarks>
/// Unsupervised meta-learning constructs pseudo-tasks by clustering gradients in a
/// low-dimensional space. Tasks within the same cluster share similar gradient structure
/// and are treated as the same "class" for self-supervised meta-training. Prediction
/// consistency regularization encourages stable cluster assignments.
/// </remarks>
public class UnsupervisedMetaLearnOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of gradient clusters (pseudo-classes) for self-supervised task construction.
    /// Default: 4.
    /// </summary>
    public int NumClusters { get; set; } = 4;

    /// <summary>
    /// Dimensionality of the compressed gradient space for clustering.
    /// Default: 32.
    /// </summary>
    public int ClusteringDim { get; set; } = 32;

    /// <summary>
    /// Weight on prediction consistency regularization between support and query
    /// adapted models. Default: 0.5.
    /// </summary>
    public double ConsistencyWeight { get; set; } = 0.5;

    /// <summary>
    /// EMA rate for updating cluster centroids. Default: 0.1.
    /// </summary>
    public double ClusterUpdateRate { get; set; } = 0.1;

    public UnsupervisedMetaLearnOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new UnsupervisedMetaLearnOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumClusters = NumClusters, ClusteringDim = ClusteringDim,
        ConsistencyWeight = ConsistencyWeight, ClusterUpdateRate = ClusterUpdateRate
    };
}
