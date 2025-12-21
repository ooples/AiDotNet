using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Almost No Inner Loop (ANIL) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// ANIL is a simplified version of MAML that only adapts the classification head
/// during inner-loop adaptation, while keeping the feature extractor frozen.
/// This significantly reduces computation while maintaining competitive performance.
/// </para>
/// <para><b>For Beginners:</b> ANIL splits a neural network into two parts:
///
/// 1. **Body (Feature Extractor):** Learns general features shared across tasks (FROZEN during adaptation)
/// 2. **Head (Classifier):** Task-specific layer that is adapted for each new task
///
/// Key insight: Most of the "learning to learn" happens in the feature extractor,
/// which doesn't need to be adapted per-task. Only the small classifier head needs
/// to change for each new task.
///
/// **Benefits:**
/// - Much faster than MAML (fewer parameters to adapt)
/// - Less memory usage (no need to store gradients for body)
/// - Often performs as well as full MAML
/// </para>
/// </remarks>
public class ANILOptions<T, TInput, TOutput> : IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a neural network with a separable body (feature extractor)
    /// and head (classifier). Only the head will be adapted during the inner loop.
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
    /// Gets or sets the optimizer for meta-parameter (outer loop) updates.
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates (head only).
    /// Default: null (uses SGD with InnerLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (head adaptation).
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-update).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps (gradient steps on support set).
    /// </summary>
    /// <value>Default is 5.</value>
    public int AdaptationSteps { get; set; } = 5;

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
    /// <value>Default is true (ANIL typically uses first-order for efficiency).</value>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region ANIL-Specific Properties

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets the dimension of the final feature representation (before head).
    /// </summary>
    /// <value>Default is 512.</value>
    public int FeatureDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to reinitialize the head for each task.
    /// </summary>
    /// <value>Default is false (use meta-learned head initialization).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, the classifier head starts fresh for each task.
    /// If false, it starts from a meta-learned initialization that is optimized across tasks.
    /// </para>
    /// </remarks>
    public bool ReinitializeHead { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use a bias term in the classification head.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseHeadBias { get; set; } = true;

    /// <summary>
    /// Gets or sets the L2 regularization strength for the head.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double HeadL2Regularization { get; set; } = 0.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the ANILOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The neural network to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public ANILOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all ANIL configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps > 0 &&
               NumClasses > 0 &&
               FeatureDimension > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0;
    }

    /// <summary>
    /// Creates a deep copy of the ANIL options.
    /// </summary>
    /// <returns>A new ANILOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new ANILOptions<T, TInput, TOutput>(MetaModel)
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
            FeatureDimension = FeatureDimension,
            ReinitializeHead = ReinitializeHead,
            UseHeadBias = UseHeadBias,
            HeadL2Regularization = HeadL2Regularization
        };
    }

    #endregion
}
