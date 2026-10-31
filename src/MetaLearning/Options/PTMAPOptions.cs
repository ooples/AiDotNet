using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// PT+MAP applies power transform normalization to features followed by MAP estimation
/// for transductive few-shot classification.
/// </para>
/// <para><b>For Beginners:</b> PT+MAP is a surprisingly simple yet powerful approach:
/// 1. Extract features using any pretrained backbone
/// 2. Apply a power transform to normalize the feature distribution
/// 3. Use MAP (optimal Bayesian) estimation to classify query examples transductively
///
/// The power transform makes features more Gaussian, which makes the simple
/// Bayesian classifier work much better. Simple math, strong results.
/// </para>
/// <para>
/// Reference: Hu, Y., Gripon, V., &amp; Pateux, S. (2021).
/// Leveraging the Feature Distribution in Transfer-based Few-Shot Learning. ICLR 2021.
/// </para>
/// </remarks>
public class PTMAPOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// <summary>Number of gradient steps taken on each task's support set (0: this algorithm adapts without gradient steps). Default: 0.</summary>
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
    /// <summary>Loss function for meta-training; null uses the default multi-class classification loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>
    /// Not used by this algorithm yet: its meta-update is built in (scaled by <see cref="OuterLearningRate"/>). A
    /// supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>
    /// Not used by this algorithm yet: any per-task adaptation is built in (scaled by <see cref="InnerLearningRate"/>).
    /// A supplied optimizer is only reset along with the learner.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Gets or sets the episodic data loader.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }
    #endregion

    #region PT+MAP-Specific Properties
    /// <summary>Gets or sets the power transform exponent (beta).</summary>
    /// <value>Default is 0.5 (square root transform).</value>
    public double PowerTransformBeta { get; set; } = 0.5;
    /// <summary>Gets or sets the number of MAP estimation iterations.</summary>
    /// <value>Default is 20.</value>
    public int MAPIterations { get; set; } = 20;
    /// <summary>Gets or sets the temperature for softmax in MAP estimation.</summary>
    /// <value>Default is 1.0.</value>
    public double Temperature { get; set; } = 1.0;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of PTMAPOptions.</summary>
    public PTMAPOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new PTMAPOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        PowerTransformBeta = PowerTransformBeta, MAPIterations = MAPIterations, Temperature = Temperature
    };
    #endregion
}
