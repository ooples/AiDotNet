using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// DKT combines deep feature extractors with Gaussian processes for Bayesian few-shot
/// classification. The deep features serve as the kernel input space, and the GP provides
/// principled uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> DKT combines neural networks with Gaussian processes:
/// 1. A neural network extracts features (like other meta-learners)
/// 2. A Gaussian Process (GP) classifier operates on these features
/// 3. The GP provides not just predictions but confidence/uncertainty estimates
/// 4. The kernel function (similarity measure) is learned end-to-end
///
/// Think of it as: the neural network finds the right feature space,
/// and the GP provides principled probabilistic classification in that space.
/// </para>
/// <para>
/// Reference: Patacchiola, M., Turner, J., Crowley, E.J., O'Boyle, M., &amp; Sherron, A. (2020).
/// Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels. ICLR 2020.
/// </para>
/// </remarks>
public class DKTOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties
    /// <summary>Gets or sets the feature extractor model.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    #endregion

    #region Standard Meta-Learning Properties
    /// <inheritdoc cref="IMetaLearnerOptions{T}.InnerLearningRate"/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.OuterLearningRate"/>
    public double OuterLearningRate { get; set; } = 0.001;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.AdaptationSteps"/>
    public int AdaptationSteps { get; set; } = 0;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.MetaBatchSize"/>
    public int MetaBatchSize { get; set; } = 4;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.NumMetaIterations"/>
    public int NumMetaIterations { get; set; } = 1000;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.GradientClipThreshold"/>
    public double? GradientClipThreshold { get; set; } = 10.0;
    /// <summary>Gets or sets the random seed.</summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationTasks"/>
    public int EvaluationTasks { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationFrequency"/>
    public int EvaluationFrequency { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EnableCheckpointing"/>
    public bool EnableCheckpointing { get; set; } = false;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.CheckpointFrequency"/>
    public int CheckpointFrequency { get; set; } = 500;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.UseFirstOrder"/>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Gets or sets the loss function.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>Gets or sets the outer loop optimizer.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>Gets or sets the inner loop optimizer.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Gets or sets the episodic data loader.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }
    #endregion

    #region DKT-Specific Properties
    /// <summary>Gets or sets the kernel type for the Gaussian process.</summary>
    /// <value>Default is "rbf" (Radial Basis Function).</value>
    public string KernelType { get; set; } = "rbf";
    /// <summary>Gets or sets the kernel length-scale parameter.</summary>
    /// <value>Default is 1.0.</value>
    public double KernelLengthScale { get; set; } = 1.0;
    /// <summary>Gets or sets the noise variance for the GP likelihood.</summary>
    /// <value>Default is 0.1.</value>
    public double NoiseVariance { get; set; } = 0.1;
    /// <summary>Gets or sets the per-example feature dimension for splitting flattened vectors.</summary>
    /// <remarks>
    /// <para>When set to 0 (default), the algorithm uses a GCD-based heuristic to estimate
    /// the per-example feature dimension from the flattened support and query vector lengths.
    /// Set this to a positive value to explicitly specify the feature dimension when the
    /// heuristic may be inaccurate (e.g., when support and query counts share common factors
    /// with the feature dimension).</para>
    /// <para><b>For Beginners:</b> This tells the GP kernel how many numbers describe each
    /// example's features. If your model outputs 64-dimensional features and you have 5
    /// support examples, the flattened vector has length 320. Setting FeatureDim = 64
    /// ensures the algorithm correctly splits it into 5 vectors of 64 dimensions each.
    /// Leave at 0 for automatic detection.</para>
    /// </remarks>
    /// <value>Default is 0 (auto-detect via GCD heuristic).</value>
    public int FeatureDim { get; set; } = 0;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of DKTOptions.</summary>
    public DKTOptions(IFullModel<T, TInput, TOutput> metaModel)
    { MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel)); }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new DKTOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        KernelType = KernelType, KernelLengthScale = KernelLengthScale, NoiseVariance = NoiseVariance,
        FeatureDim = FeatureDim
    };
    #endregion
}
