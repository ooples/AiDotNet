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

    public FreqPriorOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && LowFreqFraction > 0 && LowFreqFraction < 1;
    public IMetaLearnerOptions<T> Clone() => new FreqPriorOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        LowFreqFraction = LowFreqFraction, LowFreqRegWeight = LowFreqRegWeight, HighFreqRegWeight = HighFreqRegWeight
    };
}
