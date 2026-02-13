using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for LaplacianShot (Ziko et al., ICML 2020) few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// LaplacianShot extends nearest-centroid classification with Laplacian regularization
/// over the query set's feature graph. It encourages nearby query examples (in feature space)
/// to receive similar class assignments, propagating confident predictions to uncertain ones.
/// </para>
/// <para><b>For Beginners:</b> LaplacianShot is like "asking your neighbors for help":
///
/// **The idea:**
/// 1. Start with nearest-centroid classification (like SimpleShot)
/// 2. Build a graph connecting similar query examples
/// 3. Propagate labels through the graph: if your neighbors are confident about class A,
///    you should also lean toward class A
///
/// **Analogy:**
/// Imagine voting in a room full of people:
/// - First, everyone votes independently based on what they see
/// - Then, people discuss with their nearest neighbors
/// - After discussion, everyone updates their vote
/// - The final votes are more accurate because neighbors share information
///
/// **Why Laplacian?**
/// The "Laplacian" is a mathematical way to encode graph smoothness.
/// Laplacian regularization says: "predictions should be smooth over the graph"
/// meaning similar examples should have similar predictions.
/// </para>
/// <para>
/// Reference: Ziko, I., Dolz, J., Granger, E., &amp; Ben Ayed, I. (2020).
/// Laplacian Regularized Few-Shot Learning. ICML 2020.
/// </para>
/// </remarks>
public class LaplacianShotOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    #region LaplacianShot-Specific Properties

    /// <summary>
    /// Gets or sets the number of nearest neighbors for building the kNN graph.
    /// </summary>
    /// <value>Default is 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many "neighbors" each query example connects to
    /// in the feature graph. More neighbors = more label propagation, but too many
    /// can wash out useful local structure.
    /// </para>
    /// </remarks>
    public int KNearestNeighbors { get; set; } = 5;

    /// <summary>
    /// Gets or sets the Laplacian regularization weight.
    /// </summary>
    /// <value>Default is 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much to trust neighbor information vs. your own prediction.
    /// Higher = more smoothing (trust neighbors more). Lower = trust your own prediction more.
    /// </para>
    /// </remarks>
    public double LaplacianWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of label propagation iterations.
    /// </summary>
    /// <value>Default is 20.</value>
    public int PropagationIterations { get; set; } = 20;

    /// <summary>
    /// Gets or sets the kernel bandwidth for computing graph edge weights.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how similarity is measured between neighbors.
    /// Smaller bandwidth = only very close neighbors matter. Larger = more distant
    /// neighbors also have influence.
    /// </para>
    /// </remarks>
    public double KernelBandwidth { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the step size (alpha) for Laplacian smoothing iterations.
    /// </summary>
    /// <value>Default is 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how much predictions change per iteration.
    /// Smaller values are more stable but converge slower. Larger values converge
    /// faster but may oscillate.
    /// </para>
    /// </remarks>
    public double StepSize { get; set; } = 0.1;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of LaplacianShotOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public LaplacianShotOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
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
               KNearestNeighbors > 0 &&
               LaplacianWeight > 0 &&
               PropagationIterations > 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new LaplacianShotOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            KNearestNeighbors = KNearestNeighbors, LaplacianWeight = LaplacianWeight,
            PropagationIterations = PropagationIterations, KernelBandwidth = KernelBandwidth
        };
    }

    #endregion
}
