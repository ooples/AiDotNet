using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the DynamicTaskSampling algorithm.
/// </summary>
/// <remarks>
/// DynamicTaskSampling maintains per-task difficulty estimates and uses them to reweight
/// the meta-gradient. Tasks with higher difficulty (loss after adaptation) receive higher
/// weights via a softmax-based difficulty-proportional weighting, focusing meta-learning
/// on tasks that the model struggles with most.
/// </remarks>
public class DynamicTaskSamplingOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// EMA decay for tracking per-task difficulty (running mean of query losses).
    /// Default: 0.9.
    /// </summary>
    public double DifficultyDecay { get; set; } = 0.9;

    /// <summary>
    /// Temperature for difficulty-weighted gradient scaling. Higher values produce
    /// more uniform weights; lower values focus more on hard tasks. Default: 1.0.
    /// </summary>
    public double TaskTemperature { get; set; } = 1.0;

    /// <summary>
    /// Exploration coefficient (UCB-style) added to difficulty for exploration.
    /// Ensures under-sampled tasks still receive gradient signal. Default: 0.5.
    /// </summary>
    public double ExplorationCoeff { get; set; } = 0.5;

    public DynamicTaskSamplingOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new DynamicTaskSamplingOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        DifficultyDecay = DifficultyDecay, TaskTemperature = TaskTemperature,
        ExplorationCoeff = ExplorationCoeff
    };
}
