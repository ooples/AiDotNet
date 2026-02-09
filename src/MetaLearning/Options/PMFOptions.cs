using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for PMF (P>M>F: Pre-training, Meta-training, Fine-tuning) (Hu et al., ICLR 2022).
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// PMF introduces a three-stage training pipeline that leverages large-scale pretraining
/// before meta-learning. The key insight is that combining pre-training with meta-training
/// and optional fine-tuning achieves state-of-the-art few-shot performance.
/// </para>
/// <para><b>For Beginners:</b> PMF is a three-step recipe for few-shot learning:
///
/// **Stage 1: Pre-training (P)**
/// Train a powerful feature extractor on a large dataset (like ImageNet).
/// This gives a strong foundation of general visual knowledge.
///
/// **Stage 2: Meta-training (M)**
/// Fine-tune with episodic meta-learning on the few-shot task distribution.
/// This adapts the pretrained features for few-shot scenarios.
///
/// **Stage 3: Fine-tuning (F)**
/// Optionally fine-tune on each test task's support set for extra performance.
///
/// **Why it works:**
/// Pre-training provides rich, transferable features. Meta-training adapts them
/// for few-shot scenarios. Fine-tuning squeezes out the last bit of accuracy.
/// Each stage builds on the previous one.
/// </para>
/// <para>
/// Reference: Hu, S.X., Li, D., Stuhmer, J., Kim, M., &amp; Hospedales, T.M. (2022).
/// Pushing the Limits of Simple Pipelines for Few-Shot Learning. ICLR 2022.
/// </para>
/// </remarks>
public class PMFOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>Gets or sets the feature extractor model.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Standard Meta-Learning Properties

    /// <inheritdoc cref="IMetaLearnerOptions{T}.InnerLearningRate"/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.OuterLearningRate"/>
    public double OuterLearningRate { get; set; } = 0.0005;
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

    #region PMF-Specific Properties

    /// <summary>
    /// Gets or sets the fine-tuning learning rate for Stage 3 (F).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double FineTuningLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of fine-tuning steps during adaptation.
    /// </summary>
    /// <value>Default is 10.</value>
    public int FineTuningSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use fine-tuning during adaptation (Stage 3).
    /// </summary>
    /// <value>Default is true.</value>
    public bool EnableFineTuning { get; set; } = true;

    /// <summary>
    /// Gets or sets the distance metric for classification.
    /// </summary>
    /// <value>Default is "cosine".</value>
    public string DistanceMetric { get; set; } = "cosine";

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of PMFOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public PMFOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <inheritdoc/>
    public bool IsValid()
    {
        return MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new PMFOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            FineTuningLearningRate = FineTuningLearningRate, FineTuningSteps = FineTuningSteps,
            EnableFineTuning = EnableFineTuning, DistanceMetric = DistanceMetric
        };
    }

    #endregion
}
