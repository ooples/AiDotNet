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
    /// The number of classes the softmax classifier discriminates between - the N of N-way learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Jiang et al. eq. 6 classifies the attended context with <c>softmax(c; theta_W)</c>, so the head needs an
    /// output width. Default 5, the standard N-way few-shot setting the paper evaluates.
    /// </para>
    /// <para>
    /// <b>What went away with the rewrite.</b> AttentionDim, AttentionTemperature and AttentionEntropyWeight
    /// configured a mechanism the paper does not contain: the previous implementation bucketed the GRADIENT,
    /// softmaxed it and used the result as per-parameter learning-rate multipliers, which is Meta-SGD rather
    /// than ATAML. Eq. 5's attention is a raw inner product <c>alpha_t = theta_ATT . s_t</c> over the encoder's
    /// states - there is no softmax over it, so no temperature and no entropy to regularise.
    /// </para>
    /// </remarks>
    public int NumClasses { get; set; } = 5;

    public ATAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0 &&
        NumClasses > 0;
    public IMetaLearnerOptions<T> Clone() => new ATAMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumClasses = NumClasses
    };
}
