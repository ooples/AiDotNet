using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the ATAML (Attention-based Task-Adaptive Meta-Learning) algorithm.
/// </summary>
/// <remarks>
/// ATAML learns per-parameter attention weights that modulate the inner learning rate based
/// on the task's gradient profile. A small attention network maps compressed gradient features
/// to per-parameter scaling factors via softmax, allowing task-adaptive parameter updates.
/// </remarks>
public class ATAMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Dimension of the compressed gradient representation used as input to the attention network.
    /// Default: 32.
    /// </summary>
    public int AttentionDim { get; set; } = 32;

    /// <summary>
    /// Temperature for the attention softmax. Lower values produce sharper attention
    /// (fewer parameters get high scaling). Default: 1.0.
    /// </summary>
    public double AttentionTemperature { get; set; } = 1.0;

    /// <summary>
    /// Entropy regularization weight on the attention distribution to prevent collapse.
    /// Default: 0.01.
    /// </summary>
    public double AttentionEntropyWeight { get; set; } = 0.01;

    public ATAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new ATAMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        AttentionDim = AttentionDim, AttentionTemperature = AttentionTemperature,
        AttentionEntropyWeight = AttentionEntropyWeight
    };
}
