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
    /// <summary>Loss function for meta-training; null uses cross-entropy over logits.</summary>
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

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public ATAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null &&
        InnerLearningRate > 0 &&
        OuterLearningRate > 0 &&
        AdaptationSteps > 0 &&
        MetaBatchSize > 0 &&
        NumClasses > 0;
    /// <inheritdoc/>
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
