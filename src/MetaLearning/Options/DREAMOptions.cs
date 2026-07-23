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

    /// <summary>Hidden dimension for the reward shaper MLP. Default: 32.</summary>
    public int RewardShaperHiddenDim { get; set; } = 32;

    /// <summary>Weight for the reward shaping term in the total loss. Default: 0.5.</summary>
    public double RewardShapingWeight { get; set; } = 0.5;

    /// <summary>Discount factor for shaped reward across adaptation steps. Default: 0.99.</summary>
    public double ShapingDiscount { get; set; } = 0.99;

    public DREAMOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && RewardShaperHiddenDim > 0;
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
