using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for MCL (Meta-learning with Contrastive Learning) few-shot method.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// MCL combines episodic meta-learning with supervised contrastive learning to produce
/// features that are both discriminative and well-clustered in embedding space.
/// </para>
/// <para><b>For Beginners:</b> MCL improves features with contrastive learning:
/// 1. Standard meta-learning loss: Be good at few-shot tasks
/// 2. Contrastive loss: Same-class examples should be close, different-class far apart
/// 3. By combining both, the learned features are better organized in embedding space
///
/// Think of it as: meta-learning teaches HOW to use features for few-shot tasks,
/// while contrastive learning teaches features to BE more useful for comparison.
/// </para>
/// </remarks>
public class MCLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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

    #region MCL-Specific Properties
    /// <summary>Gets or sets the contrastive loss weight.</summary>
    /// <value>Default is 0.5.</value>
    public double ContrastiveWeight { get; set; } = 0.5;
    /// <summary>Gets or sets the temperature for contrastive loss.</summary>
    /// <value>Default is 0.07.</value>
    public double ContrastiveTemperature { get; set; } = 0.07;
    /// <summary>Gets or sets the projection dimension for contrastive head.</summary>
    /// <value>Default is 128.</value>
    public int ProjectionDim { get; set; } = 128;
    /// <summary>Gets or sets the number of ways (classes) in the few-shot task.</summary>
    /// <remarks>
    /// <para>Used to determine how many support examples belong to each class when computing
    /// supervised contrastive loss. For a 5-way 5-shot task, this would be 5.</para>
    /// <para><b>For Beginners:</b> This tells MCL how many different classes are in each
    /// training episode, so it can correctly identify which examples belong to the same
    /// class (positive pairs) vs different classes (negative pairs).</para>
    /// </remarks>
    /// <value>Default is 5 (standard 5-way few-shot setting).</value>
    public int NumWays { get; set; } = 5;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of MCLOptions.</summary>
    public MCLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel)); }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0 && ProjectionDim > 0 && ContrastiveTemperature > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new MCLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        ContrastiveWeight = ContrastiveWeight, ContrastiveTemperature = ContrastiveTemperature,
        ProjectionDim = ProjectionDim, NumWays = NumWays
    };
    #endregion
}
