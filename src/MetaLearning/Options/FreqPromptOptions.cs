using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for FreqPrompt: Frequency-domain prompt tuning for few-shot learning.
/// </summary>
/// <remarks>
/// <para>
/// FreqPrompt meta-learns additive parameter modulations ("prompts") in a frequency-domain
/// decomposition. Low-frequency prompts capture coarse domain shifts while high-frequency
/// prompts handle fine-grained task-specific adjustments. During adaptation, only the
/// prompt coefficients are updated, keeping the backbone frozen.
/// </para>
/// </remarks>
public class FreqPromptOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of frequency basis components for the prompt. Default: 16.
    /// </summary>
    public int NumFreqComponents { get; set; } = 16;

    /// <summary>
    /// Scale of prompt initialization. Default: 0.01.
    /// </summary>
    public double PromptInitScale { get; set; } = 0.01;

    /// <summary>
    /// Regularization weight for high-frequency prompt components (encourages smooth prompts). Default: 0.01.
    /// </summary>
    public double HighFreqPenalty { get; set; } = 0.01;

    public FreqPromptOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && NumFreqComponents > 0;
    public IMetaLearnerOptions<T> Clone() => new FreqPromptOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumFreqComponents = NumFreqComponents, PromptInitScale = PromptInitScale, HighFreqPenalty = HighFreqPenalty
    };
}
