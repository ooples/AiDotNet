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
    /// <summary>The model whose parameters the meta-learner trains to adapt quickly to new tasks.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    /// <inheritdoc/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc/>
    public double OuterLearningRate { get; set; } = 0.001;
    /// <inheritdoc/>
    public int AdaptationSteps { get; set; } = 5;
    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 4;
    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;
    /// <inheritdoc/>
    public double? GradientClipThreshold { get; set; } = 10.0;
    /// <inheritdoc/>
    public int? RandomSeed { get => Seed; set => Seed = value; }
    /// <inheritdoc/>
    public int EvaluationTasks { get; set; } = 100;
    /// <inheritdoc/>
    public int EvaluationFrequency { get; set; } = 100;
    /// <inheritdoc/>
    public bool EnableCheckpointing { get; set; } = false;
    /// <inheritdoc/>
    public int CheckpointFrequency { get; set; } = 500;
    /// <summary>Whether to use the first-order approximation of the meta-gradient. Default: true.</summary>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Loss function for meta-training; null uses the default multi-class classification loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>
    /// Not used by this algorithm yet: its meta-update is built in (scaled by <see cref="OuterLearningRate"/>). A
    /// supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>
    /// Not used by this algorithm yet: any per-task adaptation is built in (scaled by <see cref="InnerLearningRate"/>).
    /// A supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
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

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public HyperCLIPOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0 &&
        ProjectionDim > 0 &&
        EmbeddingDim > 0 &&
        ContrastiveTemperature > 0 &&
        ContrastiveWeight >= 0;
    /// <inheritdoc/>
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
