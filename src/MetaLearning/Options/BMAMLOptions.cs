using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for BMAML: Bayesian Model-Agnostic Meta-Learning
/// (Yoon et al., NeurIPS 2018).
/// </summary>
/// <remarks>
/// <para>
/// BMAML uses Stein Variational Gradient Descent (SVGD) to maintain a set of
/// particles (parameter vectors) that approximate the posterior over task-specific
/// parameters. Instead of a single point estimate (as in MAML), BMAML produces
/// a particle ensemble for uncertainty-aware predictions.
/// </para>
/// <para><b>Key equations:</b>
/// <code>
/// SVGD update for particle i:
///   φ_i(θ_j) = (1/M) Σ_j [k(θ_j, θ_i) ∇_{θ_j} log p(D|θ_j) + ∇_{θ_j} k(θ_j, θ_i)]
///   θ_i ← θ_i + ε * φ_i(θ_j)
///
/// RBF kernel: k(θ_i, θ_j) = exp(-||θ_i - θ_j||² / (2h²))
/// where h = median(||θ_i - θ_j||) / sqrt(log M)  (median heuristic)
/// </code>
/// </para>
/// </remarks>
public class BMAMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Number of particles (ensemble members) for SVGD posterior approximation. Default: 5.
    /// </summary>
    public int NumParticles { get; set; } = 5;

    /// <summary>
    /// Scale of Gaussian noise for initial particle perturbation from θ_0. Default: 0.01.
    /// </summary>
    public double ParticleInitScale { get; set; } = 0.01;

    /// <summary>
    /// RBF kernel bandwidth. If null, uses median heuristic: h = median(pairwise distances) / sqrt(log M). Default: null.
    /// </summary>
    public double? KernelBandwidth { get; set; }

    /// <summary>
    /// Weight for the SVGD repulsive (entropy) term relative to the attractive (likelihood) term. Default: 1.0.
    /// </summary>
    public double SVGDRepulsiveWeight { get; set; } = 1.0;

    public BMAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && NumParticles > 0 && ParticleInitScale > 0;
    public IMetaLearnerOptions<T> Clone() => new BMAMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumParticles = NumParticles, ParticleInitScale = ParticleInitScale,
        KernelBandwidth = KernelBandwidth, SVGDRepulsiveWeight = SVGDRepulsiveWeight
    };
}
