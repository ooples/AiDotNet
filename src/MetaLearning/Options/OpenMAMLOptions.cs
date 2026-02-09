using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Open-MAML (open-set MAML for open-world few-shot learning).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Open-MAML extends MAML to handle open-set scenarios where query examples may belong
/// to unknown classes not present in the support set. It adds an outlier detection mechanism
/// to MAML's adaptation process.
/// </para>
/// <para><b>For Beginners:</b> Standard few-shot assumes all queries belong to known classes.
/// Open-MAML handles the realistic case where some queries are from unknown classes:
/// 1. Adapt to support set like MAML
/// 2. Compute prototype distances for each query
/// 3. If a query is too far from ALL prototypes, classify as "unknown"
/// 4. Threshold is learned during meta-training
/// </para>
/// </remarks>
public class OpenMAMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    public int AdaptationSteps { get; set; } = 5;
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

    #region Open-MAML-Specific Properties
    /// <summary>
    /// Gets or sets the initial threshold for open-set detection.
    /// </summary>
    /// <value>Default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If a query's maximum similarity to any class is below this
    /// threshold, it's classified as "unknown." The threshold is refined during meta-training.
    /// </para>
    /// </remarks>
    public double OpenSetThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the fraction of tasks that include open-set queries during training.
    /// </summary>
    /// <value>Default is 0.3 (30% of tasks have open-set examples).</value>
    public double OpenSetTaskFraction { get; set; } = 0.3;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of OpenMAMLOptions.</summary>
    public OpenMAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel)); }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new OpenMAMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        OpenSetThreshold = OpenSetThreshold, OpenSetTaskFraction = OpenSetTaskFraction
    };
    #endregion
}
