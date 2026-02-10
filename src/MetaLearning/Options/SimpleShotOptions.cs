using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for SimpleShot (Wang et al., 2019) few-shot classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// SimpleShot demonstrates that nearest-centroid classification with proper feature
/// normalization (L2 or CL2N) can match or exceed many complex meta-learning methods.
/// It requires no task-specific adaptation - just normalize features and compare distances.
/// </para>
/// <para><b>For Beginners:</b> SimpleShot shows that simple methods can be surprisingly effective:
///
/// **The approach:**
/// 1. Train a good feature extractor on the base classes (standard training, no episodes)
/// 2. For a new few-shot task:
///    - Extract features for all support and query examples
///    - Normalize the features (L2 norm or centered L2 norm)
///    - Compute class centroids from support features
///    - Classify query examples by nearest centroid
///
/// **Why it works:**
/// Many complex meta-learning methods spend effort on task-specific adaptation, but
/// a well-trained feature extractor with proper normalization already produces features
/// where nearest-centroid works well.
///
/// **Normalization methods:**
/// - L2: Normalize each feature vector to unit length
/// - CL2N: Center features (subtract mean), then L2 normalize
/// - CL2N typically works better because it removes the "bias" in feature space
/// </para>
/// <para>
/// Reference: Wang, Y., Chao, W.L., Weinberger, K.Q., &amp; van der Maaten, L. (2019).
/// SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning.
/// </para>
/// </remarks>
public class SimpleShotOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the feature extractor model.
    /// </summary>
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
    /// <summary>Gets or sets the inner loop optimizer (unused for SimpleShot).</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Gets or sets the episodic data loader.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    #endregion

    #region SimpleShot-Specific Properties

    /// <summary>
    /// Gets or sets the feature normalization method.
    /// </summary>
    /// <value>Default is "CL2N" (centered L2 normalization).</value>
    /// <remarks>
    /// <para>
    /// Available methods:
    /// - "L2": L2 normalization (unit vector)
    /// - "CL2N": Centered L2 normalization (center then normalize)
    /// - "None": No normalization (raw features)
    /// </para>
    /// <para><b>For Beginners:</b> How to prepare features before comparing them:
    /// - "L2": Make all feature vectors the same length (unit vectors)
    /// - "CL2N": First center features (subtract the average), then make them unit length
    /// - CL2N typically works best because it removes feature bias
    /// </para>
    /// </remarks>
    public string NormalizationType { get; set; } = "CL2N";

    /// <summary>
    /// Gets or sets the distance metric for nearest-centroid classification.
    /// </summary>
    /// <value>Default is "cosine".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to measure similarity between features:
    /// - "cosine": Angle between feature vectors (good for high-dimensional features)
    /// - "euclidean": Straight-line distance (good for low-dimensional features)
    /// </para>
    /// </remarks>
    public string DistanceMetric { get; set; } = "cosine";

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of SimpleShotOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public SimpleShotOptions(IFullModel<T, TInput, TOutput> metaModel)
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
               NumMetaIterations > 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new SimpleShotOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            NormalizationType = NormalizationType, DistanceMetric = DistanceMetric
        };
    }

    #endregion
}
