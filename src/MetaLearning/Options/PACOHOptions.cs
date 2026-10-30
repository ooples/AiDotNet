using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters
/// (Rothfuss et al., ICLR 2021).
/// </summary>
/// <remarks>
/// <para>
/// PACOH meta-learns a Gaussian prior N(μ, σ²I) over neural network parameters that
/// provides PAC-Bayesian generalization guarantees. The outer loop optimizes the prior
/// to minimize a PAC-Bayesian bound; the inner loop performs MAP estimation with
/// the learned prior as regularizer.
/// </para>
/// </remarks>
public class PACOHOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// KL divergence coefficient in the PAC-Bayesian bound. Controls strength of prior
    /// regularization. Derived from (ln(2√n/δ)) / n in the theory. Default: 0.1.
    /// </summary>
    public double KLCoefficient { get; set; } = 0.1;

    /// <summary>
    /// Initial log-variance of the prior distribution. Default: -3.0 (σ² ≈ 0.05).
    /// </summary>
    public double InitialLogVariance { get; set; } = -3.0;

    /// <summary>
    /// Number of posterior samples for Monte Carlo estimation of expected loss. Default: 5.
    /// </summary>
    public int NumPosteriorSamples { get; set; } = 5;

    /// <summary>
    /// Confidence parameter δ for the PAC-Bayesian bound (higher = tighter but more conservative). Default: 0.05.
    /// </summary>
    public double Delta { get; set; } = 0.05;

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public PACOHOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && KLCoefficient > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new PACOHOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        KLCoefficient = KLCoefficient, InitialLogVariance = InitialLogVariance,
        NumPosteriorSamples = NumPosteriorSamples, Delta = Delta
    };
}
