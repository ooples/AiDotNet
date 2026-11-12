using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// The metric PMF's ProtoNet classifier scores a query against the class prototypes with.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each class is summarized by the mean feature vector of its support examples (its
/// prototype). The metric decides how a query is compared with each prototype: by the angle between them (cosine)
/// or by their straight-line distance (squared Euclidean).
/// </para>
/// </remarks>
public enum PMFDistanceMetric
{
    /// <summary>
    /// Cosine similarity scaled by a learnable inverse temperature, <c>cos(f(x), c_k) / tau</c>: the paper's choice
    /// (Hu et al. 2022, following Chen et al. 2021).
    /// </summary>
    Cosine,

    /// <summary>
    /// Negative squared Euclidean distance, <c>-||f(x) - c_k||^2</c>, as in the original ProtoNet (Snell et al. 2017).
    /// </summary>
    SquaredEuclidean
}

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
/// Fine-tune the feature extractor episodically as a ProtoNet with a cosine classifier: each class's
/// prototype is the mean of its support features, and a query's logits are its cosine similarity to
/// each prototype divided by a learnable temperature. This adapts the pretrained features for few-shot scenarios.
///
/// **Stage 3: Fine-tuning (F)**
/// Optionally fine-tune on each test task's support set, with the same ProtoNet loss, for extra performance.
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
    /// <summary>Learning rate of the outer (meta) update. Default: 0.0005.</summary>
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
    /// <summary>Loss function for meta-training; null uses the default multi-class classification loss.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>
    /// Optimizer for the meta-update. When set, it takes the outer step at its own learning rate (so
    /// <see cref="OuterLearningRate"/> no longer applies) and keeps its state across meta-iterations; null uses
    /// plain SGD at <see cref="OuterLearningRate"/>. It must not be the same instance as <see cref="InnerOptimizer"/>.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>
    /// Optimizer for task adaptation. When set, it takes each inner step at its own learning rate (so
    /// <see cref="InnerLearningRate"/> no longer applies) and is reset at the start of every adaptation; null uses
    /// plain SGD at <see cref="InnerLearningRate"/>.
    /// It is rejected when the configuration uses a separate fine-tuning learning rate, which assumes plain SGD inner steps.
    /// </summary>
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
    /// Gets or sets the metric the ProtoNet classifier of Stage 2 (M) and of adaptation scores queries with.
    /// </summary>
    /// <value>Default is <see cref="PMFDistanceMetric.Cosine"/>, the paper's cosine classifier.</value>
    public PMFDistanceMetric DistanceMetric { get; set; } = PMFDistanceMetric.Cosine;

    /// <summary>
    /// Gets or sets the initial temperature of the cosine classifier: the logits are
    /// <c>cos(f(x), c_k) / Temperature</c>.
    /// </summary>
    /// <value>
    /// Default is 0.1, the reference implementation's cosine scale of 10 (logits <c>10 * cos</c>). Ignored by
    /// <see cref="PMFDistanceMetric.SquaredEuclidean"/>.
    /// </value>
    public double Temperature { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether the cosine classifier's scale <c>1 / Temperature</c> is learned during meta-training, as
    /// in the paper's ProtoNet, or kept fixed at its initial value.
    /// </summary>
    /// <value>Default is true.</value>
    public bool LearnTemperature { get; set; } = true;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of PMFOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public PMFOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <inheritdoc/>
    public bool IsValid()
    {
        return MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0 && InnerLearningRate > 0
            && Temperature > 0 && !double.IsNaN(Temperature) && !double.IsInfinity(Temperature);
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
            EnableFineTuning = EnableFineTuning, DistanceMetric = DistanceMetric,
            Temperature = Temperature, LearnTemperature = LearnTemperature
        };
    }

    #endregion
}
