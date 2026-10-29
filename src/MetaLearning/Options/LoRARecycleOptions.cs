using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for LoRA-Recycle (Hu et al., CVPR 2025).
/// </summary>
/// <remarks>
/// <para>
/// LoRA-Recycle distills a "meta-LoRA" from diverse pre-tuned LoRA adapters without
/// accessing their private training data. It supports tuning-free few-shot adaptation
/// by recycling knowledge from previously-learned tasks via prototype-based matching.
/// </para>
/// <para><b>Key Parameters:</b>
/// <list type="bullet">
/// <item><see cref="NumRecycledAdapters"/> — number of previously-learned LoRA adapters to maintain</item>
/// <item><see cref="Rank"/> — rank of each LoRA adapter</item>
/// <item><see cref="PrototypeDim"/> — dimensionality of prototype embeddings for adapter selection</item>
/// </list>
/// </para>
/// </remarks>
public class LoRARecycleOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// <inheritdoc/>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Loss function used in the inner and outer loops; null uses the meta-model's default loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>Optimizer for the outer (meta) update; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>Optimizer for the inner, per-task adaptation loop; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Rank of each LoRA adapter (controls adapter capacity). Default: 4.
    /// </summary>
    public int Rank { get; set; } = 4;

    /// <summary>
    /// Number of recycled LoRA adapters to maintain in the adapter bank. Default: 5.
    /// </summary>
    public int NumRecycledAdapters { get; set; } = 5;

    /// <summary>
    /// Dimensionality of the prototype embedding used for adapter selection. Default: 32.
    /// </summary>
    public int PrototypeDim { get; set; } = 32;

    /// <summary>
    /// Temperature parameter for softmax-based adapter weighting. Default: 1.0.
    /// </summary>
    public double SelectionTemperature { get; set; } = 1.0;

    /// <summary>
    /// KL divergence weight for distillation loss. Default: 0.005.
    /// </summary>
    public double KLWeight { get; set; } = 0.005;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public LoRARecycleOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 &&
        Rank > 0 && NumRecycledAdapters > 0 && PrototypeDim > 0 && SelectionTemperature > 0 && KLWeight >= 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new LoRARecycleOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        Rank = Rank, NumRecycledAdapters = NumRecycledAdapters, PrototypeDim = PrototypeDim,
        SelectionTemperature = SelectionTemperature, KLWeight = KLWeight
    };
}
