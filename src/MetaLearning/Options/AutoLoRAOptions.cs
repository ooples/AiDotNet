using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for AutoLoRA (Zhang et al., NAACL 2024).
/// </summary>
/// <remarks>
/// <para>
/// AutoLoRA automatically determines optimal per-layer LoRA ranks via meta-learning.
/// Each rank-1 component has a continuous selection variable α ∈ [0,1] optimized on
/// validation data (outer loop), while LoRA weights are trained on training data (inner loop).
/// Final rank is determined by thresholding: k_l = |{α_{l,j} | α_{l,j} ≥ λ}|.
/// </para>
/// </remarks>
public class AutoLoRAOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    /// <summary>The model whose parameters the meta-learner trains to adapt quickly to new tasks.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    /// <inheritdoc/>
    public double InnerLearningRate { get; set; } = 0.0001;
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
    /// Maximum rank per group (initial rank allocation). Default: 8.
    /// </summary>
    public int MaxRank { get; set; } = 8;

    /// <summary>
    /// Number of rank groups (analogous to layers). Each group gets independent rank selection.
    /// Default: 4.
    /// </summary>
    public int NumRankGroups { get; set; } = 4;

    /// <summary>
    /// Threshold for rank determination: α_{l,j} ≥ threshold means rank-1 component is kept.
    /// Default: 0.125.
    /// </summary>
    public double RankThreshold { get; set; } = 0.125;

    /// <summary>
    /// Regularization penalty that encourages lower ranks (sparsity). Default: 0.01.
    /// </summary>
    public double RankRegularization { get; set; } = 0.01;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public AutoLoRAOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && MaxRank > 0 && NumRankGroups > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new AutoLoRAOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        MaxRank = MaxRank, NumRankGroups = NumRankGroups, RankThreshold = RankThreshold,
        RankRegularization = RankRegularization
    };
}
