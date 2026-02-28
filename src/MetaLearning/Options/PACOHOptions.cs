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

    public PACOHOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && KLCoefficient > 0;
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
