using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for ConstellationNet (Xu et al., ICLR 2021).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// ConstellationNet learns structured representations where parts and their spatial
/// relationships form "constellations" for few-shot classification.
/// </para>
/// <para><b>For Beginners:</b> ConstellationNet finds patterns in how parts relate:
/// 1. Detect discriminative parts/regions in each example
/// 2. Learn how these parts are arranged (their "constellation")
/// 3. Compare queries to support classes by matching constellation patterns
///
/// Imagine recognizing a face: you don't just look at eyes, nose, mouth separately.
/// You also look at HOW they're arranged - that's the constellation.
/// </para>
/// <para>
/// Reference: Xu, C., Fu, Y., Liu, C., Wang, C., Li, J., Huang, F., Zhang, L., &amp; Xue, X. (2021).
/// Learning Dynamic Alignment via Meta-filter for Few-shot Learning. CVPR 2021.
/// </para>
/// </remarks>
public class ConstellationNetOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    #region ConstellationNet-Specific Properties
    /// <summary>Gets or sets the number of constellation parts to detect.</summary>
    /// <value>Default is 8.</value>
    public int NumParts { get; set; } = 8;
    /// <summary>Gets or sets the part feature dimension.</summary>
    /// <value>Default is 64.</value>
    public int PartFeatureDim { get; set; } = 64;
    /// <summary>Gets or sets whether to use spatial relationships between parts.</summary>
    /// <value>Default is true.</value>
    public bool UseSpatialRelations { get; set; } = true;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of ConstellationNetOptions.</summary>
    public ConstellationNetOptions(IFullModel<T, TInput, TOutput> metaModel)
    { MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel)); }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new ConstellationNetOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumParts = NumParts, PartFeatureDim = PartFeatureDim, UseSpatialRelations = UseSpatialRelations
    };
    #endregion
}
