using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for DREAM: Directed REward Augmented Meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// DREAM augments MAML-style meta-learning with a learned reward/loss shaping function
/// that transforms the raw task loss into a more informative gradient signal. The reward
/// shaper maps (loss, gradient_norm, step) → shaped_loss, enabling the inner loop
/// to receive curriculum-like guidance that accelerates adaptation.
/// </para>
/// </remarks>
public class DREAMOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    /// <summary>Hidden dimension for the reward shaper MLP. Default: 32.</summary>
    public int RewardShaperHiddenDim { get; set; } = 32;

    /// <summary>Weight for the reward shaping term in the total loss. Default: 0.5.</summary>
    public double RewardShapingWeight { get; set; } = 0.5;

    /// <summary>Discount factor for shaped reward across adaptation steps. Default: 0.99.</summary>
    public double ShapingDiscount { get; set; } = 0.99;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public DREAMOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && RewardShaperHiddenDim > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new DREAMOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        RewardShaperHiddenDim = RewardShaperHiddenDim, RewardShapingWeight = RewardShapingWeight,
        ShapingDiscount = ShapingDiscount
    };
}
