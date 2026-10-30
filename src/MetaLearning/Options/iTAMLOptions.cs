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
    /// <summary>Loss function for meta-training; null uses the default multi-class classification loss.</summary>
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

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public iTAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() =>
        MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 &&
        TeacherEmaDecay > 0 && TeacherEmaDecay < 1 && DistillationWeight >= 0 &&
        DistillationTemperature > 0;
    /// <inheritdoc/>
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
