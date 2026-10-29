using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for FreqPrior: Frequency-based prior for cross-domain few-shot learning.
/// </summary>
/// <remarks>
/// <para>
/// FreqPrior decomposes the parameter space into frequency bands using a discrete cosine-like
/// transform. Low-frequency components capture domain-invariant structure and are strongly
/// regularized toward the meta-learned prior, while high-frequency components are allowed to
/// vary freely for task-specific adaptation. This frequency-based prior encourages learning
/// smooth, transferable representations.
/// </para>
/// </remarks>
public class FreqPriorOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Fraction of parameters considered "low frequency" (strongly regularized). Default: 0.3.
    /// </summary>
    public double LowFreqFraction { get; set; } = 0.3;

    /// <summary>
    /// Regularization strength for low-frequency components toward the meta-prior. Default: 1.0.
    /// </summary>
    public double LowFreqRegWeight { get; set; } = 1.0;

    /// <summary>
    /// Regularization strength for high-frequency components (allows more variation). Default: 0.01.
    /// </summary>
    public double HighFreqRegWeight { get; set; } = 0.01;

    /// <summary>
    /// Scale factor for the frequency penalty term applied to low-frequency gradient variance.
    /// Controls how strongly cross-task consistency of low-frequency gradients is enforced. Default: 0.01.
    /// </summary>
    public double FrequencyPenaltyScale { get; set; } = 0.01;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public FreqPriorOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && LowFreqFraction > 0 && LowFreqFraction < 1 && LowFreqRegWeight >= 0 && HighFreqRegWeight >= 0
        && FrequencyPenaltyScale >= 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new FreqPriorOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        LowFreqFraction = LowFreqFraction, LowFreqRegWeight = LowFreqRegWeight, HighFreqRegWeight = HighFreqRegWeight,
        FrequencyPenaltyScale = FrequencyPenaltyScale
    };
}
