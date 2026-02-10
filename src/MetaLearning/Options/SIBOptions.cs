using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for SIB (Sequential Information Bottleneck) (Hu et al., 2020) few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// SIB uses the information bottleneck principle for transductive few-shot learning.
/// It iteratively optimizes cluster assignments by maximizing mutual information between
/// data representations and cluster labels while compressing nuisance information.
/// </para>
/// <para><b>For Beginners:</b> SIB balances two competing goals:
///
/// **The Information Bottleneck principle:**
/// 1. Keep USEFUL information: Cluster assignments should predict labels well
/// 2. Remove USELESS information: Don't memorize irrelevant details
///
/// **How SIB works for few-shot learning:**
/// 1. Initialize cluster centroids from support class prototypes
/// 2. Assign ALL examples (support + query) to clusters
/// 3. Iteratively refine assignments by:
///    - Reassigning each example to the most informative cluster
///    - Updating cluster centroids based on new assignments
///    - Balancing information retention vs. compression
///
/// **Analogy: Efficient note-taking**
/// Imagine taking notes from a lecture:
/// - Too much detail = you copy everything (no compression, hard to review)
/// - Too little detail = you miss key points (too much compression)
/// - SIB finds the sweet spot: keep important info, discard noise
///
/// **Key property:** Transductive - uses query set structure for better predictions
/// </para>
/// <para>
/// Reference: Hu, Y., Gripon, V., &amp; Pateux, S. (2020).
/// Leveraging the Feature Distribution in Transfer-based Few-Shot Learning.
/// </para>
/// </remarks>
public class SIBOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    public int AdaptationSteps { get; set; } = 1;
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

    #region SIB-Specific Properties

    /// <summary>
    /// Gets or sets the number of SIB optimization iterations.
    /// </summary>
    /// <value>Default is 30.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many rounds of cluster refinement to perform.
    /// More iterations = better clusters but slower. 30 is usually enough for convergence.
    /// </para>
    /// </remarks>
    public int NumSIBIterations { get; set; } = 30;

    /// <summary>
    /// Gets or sets the number of random restarts for avoiding local optima.
    /// </summary>
    /// <value>Default is 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many times to run SIB from different starting points
    /// and pick the best result. More restarts = more likely to find the global optimum.
    /// </para>
    /// </remarks>
    public int NumRestarts { get; set; } = 3;

    /// <summary>
    /// Gets or sets the compression parameter (beta) for the information bottleneck.
    /// </summary>
    /// <value>Default is 10.0.</value>
    /// <remarks>
    /// <para>
    /// Higher beta = more information retained (less compression).
    /// Lower beta = more compression (may lose useful info).
    /// </para>
    /// <para><b>For Beginners:</b> Controls the information-compression tradeoff.
    /// High beta: keep more information (safer, less risk of losing important patterns).
    /// Low beta: compress more (faster, but may throw away useful details).
    /// </para>
    /// </remarks>
    public double Beta { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the temperature for soft cluster assignments.
    /// </summary>
    /// <value>Default is 1.0.</value>
    public double Temperature { get; set; } = 1.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of SIBOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public SIBOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <inheritdoc/>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               NumSIBIterations > 0 &&
               Beta > 0 &&
               Temperature > 0 &&
               NumRestarts > 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new SIBOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            NumSIBIterations = NumSIBIterations, NumRestarts = NumRestarts,
            Beta = Beta, Temperature = Temperature
        };
    }

    #endregion
}
