using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning
/// with per-group prior variances.
/// </summary>
/// <remarks>
/// <para>
/// Meta-PACOH extends PACOH by introducing a hierarchical Bayesian structure where
/// different parameter groups (e.g., layers) have independently learned prior variances.
/// This enables the algorithm to express that some parameter groups should be more
/// tightly constrained to the meta-learned prior while others can vary freely.
/// </para>
/// <para>
/// The algorithm meta-learns both a shared prior mean μ_P and per-group prior
/// log-variances {log(σ²_g)} to minimize a hierarchical PAC-Bayesian bound.
/// </para>
/// </remarks>
public class MetaPACOHOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of parameter groups with independent prior variances. Default: 4.
    /// Groups are formed by evenly partitioning the parameter vector.
    /// </summary>
    public int NumPriorGroups { get; set; } = 4;

    /// <summary>
    /// Initial log-variance for all prior groups. Default: -3.0 (σ² ≈ 0.05).
    /// </summary>
    public double InitialLogVariance { get; set; } = -3.0;

    /// <summary>
    /// KL divergence coefficient for the hierarchical PAC-Bayesian bound. Default: 0.1.
    /// </summary>
    public double KLCoefficient { get; set; } = 0.1;

    /// <summary>
    /// Confidence parameter δ for the PAC-Bayesian bound. Default: 0.05.
    /// </summary>
    public double Delta { get; set; } = 0.05;

    /// <summary>
    /// Hyper-prior log-variance controlling how much per-group variances can deviate. Default: 0.0 (σ² = 1.0).
    /// </summary>
    public double HyperPriorLogVar { get; set; } = 0.0;

    public MetaPACOHOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && NumPriorGroups > 0 && KLCoefficient > 0;
    public IMetaLearnerOptions<T> Clone() => new MetaPACOHOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumPriorGroups = NumPriorGroups, InitialLogVariance = InitialLogVariance,
        KLCoefficient = KLCoefficient, Delta = Delta, HyperPriorLogVar = HyperPriorLogVar
    };
}
