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
    /// <inheritdoc/>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Loss function used in the inner and outer loops; null uses the meta-model's default loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>Optimizer for the outer (meta) update; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>Optimizer for the inner, per-task adaptation loop; null uses the algorithm's default.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Episodic data loader that supplies the meta-training tasks; null when tasks are passed to training directly.</summary>
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

    /// <summary>Creates options for meta-learning <paramref name="metaModel"/>, with the default settings.</summary>
    /// <param name="metaModel">The model whose parameters are meta-learned.</param>
    public BMAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }

    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0
        && NumParticles > 0 && ParticleInitScale > 0 && SVGDRepulsiveWeight >= 0;
    /// <inheritdoc/>
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
