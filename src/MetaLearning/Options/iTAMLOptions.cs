using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the iTAML (incremental Task-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <remarks>
/// <para>
/// iTAML (Rajasegaran et al., 2020) prevents catastrophic forgetting by maintaining an EMA
/// teacher model and applying knowledge distillation between teacher and student predictions.
/// Task-balanced gradient weighting normalizes gradient magnitudes across tasks to prevent
/// any single task from dominating the meta-update.
/// </para>
/// </remarks>
public class iTAMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// EMA decay rate for the teacher model. Values closer to 1.0 make the teacher
    /// more stable. Default: 0.999.
    /// </summary>
    public double TeacherEmaDecay { get; set; } = 0.999;

    /// <summary>
    /// Weight of the knowledge distillation loss between student and teacher predictions.
    /// L_total = L_task + DistillationWeight * L_distill. Default: 1.0.
    /// </summary>
    public double DistillationWeight { get; set; } = 1.0;

    /// <summary>
    /// Temperature for softening predictions in the distillation loss.
    /// Higher values produce softer probability distributions. Default: 2.0.
    /// </summary>
    public double DistillationTemperature { get; set; } = 2.0;

    /// <summary>
    /// Whether to normalize gradient magnitudes across tasks to prevent
    /// high-loss tasks from dominating. Default: true.
    /// </summary>
    public bool TaskBalancingEnabled { get; set; } = true;

    public iTAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0;
    public IMetaLearnerOptions<T> Clone() => new iTAMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        TeacherEmaDecay = TeacherEmaDecay, DistillationWeight = DistillationWeight,
        DistillationTemperature = DistillationTemperature, TaskBalancingEnabled = TaskBalancingEnabled
    };
}
