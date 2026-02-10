using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FRN classifies queries by reconstructing their feature maps from class-specific support
/// feature maps. The reconstruction error serves as the distance metric.
/// </para>
/// <para><b>For Beginners:</b> FRN asks "can I rebuild this query from the support set?":
/// 1. For each class, collect support feature maps
/// 2. Try to reconstruct the query's feature map using ONLY features from class k
/// 3. If reconstruction is good (low error): query likely belongs to class k
/// 4. If reconstruction is bad (high error): query likely NOT class k
///
/// It's like asking: "Can I explain this new image using only the patterns I've seen
/// from cats?" If yes, it's probably a cat. If not, try the next class.
/// </para>
/// <para>
/// Reference: Wertheimer, D., Tang, L., &amp; Hariharan, B. (2021).
/// Few-Shot Classification With Feature Map Reconstruction Networks. CVPR 2021.
/// </para>
/// </remarks>
public class FRNOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    #region FRN-Specific Properties
    /// <summary>Gets or sets the reconstruction regularization parameter (lambda).</summary>
    /// <value>Default is 0.01. Prevents degenerate reconstruction solutions.</value>
    public double ReconstructionLambda { get; set; } = 0.01;
    /// <summary>Gets or sets the number of reconstruction components per class.</summary>
    /// <value>Default is 5 (one per support example in 5-shot).</value>
    public int NumComponents { get; set; } = 5;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of FRNOptions.</summary>
    public FRNOptions(IFullModel<T, TInput, TOutput> metaModel)
    { MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel)); }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new FRNOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        ReconstructionLambda = ReconstructionLambda, NumComponents = NumComponents
    };
    #endregion
}
