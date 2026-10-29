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

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public MetaPACOHOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && NumPriorGroups > 0 && KLCoefficient > 0;
    /// <inheritdoc/>
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
