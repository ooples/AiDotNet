using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Task-Dependent Adaptive Metric (TADAM) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// TADAM extends prototypical networks by incorporating task-dependent metric learning.
/// It uses task conditioning (TC) to modulate the feature extraction process based on
/// the task at hand, and metric scaling to adapt distances in embedding space.
/// </para>
/// <para><b>For Beginners:</b> TADAM improves on Prototypical Networks by:
///
/// 1. **Task Conditioning (TC):** Adjusts features based on the specific task
/// 2. **Metric Scaling:** Learns how to weight different feature dimensions
/// 3. **Auxiliary Co-Training:** Uses additional classification to improve features
///
/// Think of it as ProtoNets that "pay attention" to what matters for each specific task.
/// </para>
/// </remarks>
public class TADAMOptions<T, TInput, TOutput> : IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (feature encoder) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a feature encoder that maps inputs to an embedding space.
    /// TADAM will apply task conditioning to modulate the extracted features.
    /// </para>
    /// <para><b>For Beginners:</b> This is typically a CNN for images. TADAM will learn
    /// to adjust how this encoder works based on each specific task.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses cross-entropy loss internally).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for network updates.
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates.
    /// Default: null (TADAM uses feed-forward comparison with task conditioning).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (not used in TADAM).
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (encoder training).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps.
    /// </summary>
    /// <value>Default is 1 (TADAM uses feed-forward task conditioning).</value>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default is 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default is 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate the meta-learner.
    /// </summary>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save model checkpoints.
    /// </summary>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    public int CheckpointFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default is true since TADAM doesn't use gradient-based inner loop.</value>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region TADAM-Specific Properties

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use task conditioning (FiLM layers).
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Task conditioning adjusts the feature encoder based on
    /// what task is being solved. This helps the network adapt its representation.
    /// </para>
    /// </remarks>
    public bool UseTaskConditioning { get; set; } = true;

    /// <summary>
    /// Gets or sets the dimension of task embeddings.
    /// </summary>
    /// <value>Default is 64.</value>
    public int TaskEmbeddingDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets whether to use metric scaling.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Metric scaling learns to weight different dimensions
    /// of the embedding space differently when computing distances.
    /// </para>
    /// </remarks>
    public bool UseMetricScaling { get; set; } = true;

    /// <summary>
    /// Gets or sets the initial value for the learnable temperature parameter.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// <para>
    /// The temperature controls how "soft" or "hard" the distance-based classification is.
    /// Lower values make the softmax sharper (more confident predictions).
    /// </para>
    /// </remarks>
    public double InitialTemperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use auxiliary co-training.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Auxiliary co-training adds an extra classification
    /// objective to help learn better features. It can improve performance but adds
    /// computational cost.
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryCoTraining { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for auxiliary loss.
    /// </summary>
    /// <value>Default is 0.5.</value>
    public double AuxiliaryLossWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    /// <value>Default is 512.</value>
    public int EmbeddingDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to normalize embeddings.
    /// </summary>
    /// <value>Default is true.</value>
    public bool NormalizeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets the L2 regularization strength.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double L2Regularization { get; set; } = 0.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the TADAMOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The feature encoder to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public TADAMOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all TADAM configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               NumClasses > 0 &&
               EmbeddingDimension > 0 &&
               TaskEmbeddingDimension > 0 &&
               InitialTemperature > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0;
    }

    /// <summary>
    /// Creates a deep copy of the TADAM options.
    /// </summary>
    /// <returns>A new TADAMOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new TADAMOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer,
            DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate,
            OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps,
            MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            UseFirstOrder = UseFirstOrder,
            NumClasses = NumClasses,
            UseTaskConditioning = UseTaskConditioning,
            TaskEmbeddingDimension = TaskEmbeddingDimension,
            UseMetricScaling = UseMetricScaling,
            InitialTemperature = InitialTemperature,
            UseAuxiliaryCoTraining = UseAuxiliaryCoTraining,
            AuxiliaryLossWeight = AuxiliaryLossWeight,
            EmbeddingDimension = EmbeddingDimension,
            NormalizeEmbeddings = NormalizeEmbeddings,
            L2Regularization = L2Regularization
        };
    }

    #endregion
}
