using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for SDCL: Self-Distillation Collaborative Learning for meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// SDCL applies self-distillation within the meta-learning framework. A teacher model
/// (exponential moving average of the student) provides soft targets that regularize
/// the adapted student's predictions. The KL divergence between teacher and student
/// predictions acts as a collaborative learning signal that stabilizes adaptation
/// and improves cross-domain generalization.
/// </para>
/// </remarks>
public class SDCLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// EMA decay rate for the teacher model. Default: 0.999.
    /// </summary>
    public double TeacherEmaDecay { get; set; } = 0.999;

    /// <summary>
    /// Weight for the distillation (KL divergence) loss. Default: 0.5.
    /// </summary>
    public double DistillationWeight { get; set; } = 0.5;

    /// <summary>
    /// Temperature for softening teacher/student predictions. Default: 4.0.
    /// </summary>
    public double DistillationTemperature { get; set; } = 4.0;

    public SDCLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && TeacherEmaDecay > 0 && TeacherEmaDecay < 1 && DistillationTemperature > 0;
    public IMetaLearnerOptions<T> Clone() => new SDCLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        TeacherEmaDecay = TeacherEmaDecay, DistillationWeight = DistillationWeight,
        DistillationTemperature = DistillationTemperature
    };
}
