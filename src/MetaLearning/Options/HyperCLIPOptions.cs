using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>Configuration options for HyperCLIP meta-learning.</summary>
/// <remarks>
/// <para>
/// HyperCLIP uses contrastive alignment between task embeddings (from support gradients)
/// and parameter embeddings (from adapted parameters). An InfoNCE-style contrastive loss
/// ensures that a task's embedding is closest to its own adapted parameter embedding,
/// enabling zero-shot parameter generation for new tasks.
/// </para>
/// </remarks>
public class HyperCLIPOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Dimension of the shared projection space for contrastive alignment.
    /// Both task and parameter embeddings are projected to this dimension.
    /// Default: 32.
    /// </summary>
    public int ProjectionDim { get; set; } = 32;

    /// <summary>
    /// Temperature for the InfoNCE contrastive loss (τ in exp(sim/τ)).
    /// Lower values produce sharper distributions. Default: 0.07.
    /// </summary>
    public double ContrastiveTemperature { get; set; } = 0.07;

    /// <summary>
    /// Weight of the contrastive alignment loss relative to the task loss.
    /// L_total = L_query + ContrastiveWeight * L_contrastive.
    /// Default: 0.1.
    /// </summary>
    public double ContrastiveWeight { get; set; } = 0.1;

    /// <summary>
    /// Gradient compression dimension for computing embeddings from gradients.
    /// Default: 32.
    /// </summary>
    public int EmbeddingDim { get; set; } = 32;

    public HyperCLIPOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
                              && ProjectionDim > 0 && ContrastiveTemperature > 0;
    public IMetaLearnerOptions<T> Clone() => new HyperCLIPOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        ProjectionDim = ProjectionDim, ContrastiveTemperature = ContrastiveTemperature,
        ContrastiveWeight = ContrastiveWeight, EmbeddingDim = EmbeddingDim
    };
}
