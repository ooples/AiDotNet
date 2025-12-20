using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Body Only Inner Loop (BOIL) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// BOIL is the opposite of ANIL - it only adapts the feature extractor (body) during
/// inner-loop adaptation while keeping the classification head frozen. This explores
/// the hypothesis that task-specific features are more important than task-specific classifiers.
/// </para>
/// <para><b>For Beginners:</b> BOIL splits a neural network into two parts:
/// </para>
/// <list type="number">
/// <item><b>Body (Feature Extractor):</b> ADAPTED for each new task</item>
/// <item><b>Head (Classifier):</b> FROZEN during adaptation (uses meta-learned weights)</item>
/// </list>
/// <para>
/// This is the opposite of ANIL (which freezes body, adapts head). BOIL tests whether
/// it's better to adapt HOW we see things rather than HOW we decide.
/// </para>
/// <para>
/// <b>When to use BOIL:</b>
/// - When tasks differ more in their visual/input patterns than their decision boundaries
/// - When you have a good meta-learned classifier that works across tasks
/// - When you want to experiment with different adaptation strategies
/// </para>
/// <para>
/// Reference: Oh, J., Yoo, H., Kim, C., &amp; Yun, S. Y. (2021).
/// BOIL: Towards Representation Change for Few-shot Learning.
/// </para>
/// </remarks>
public class BOILOptions<T, TInput, TOutput> : IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a neural network with a separable body (feature extractor)
    /// and head (classifier). Only the body will be adapted during the inner loop.
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
    /// Gets or sets the optimizer for inner loop updates (body only).
    /// Default: null (uses SGD with InnerLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (body adaptation).
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
    /// <value>Default is true (BOIL typically uses first-order for efficiency).</value>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region BOIL-Specific Properties

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
    /// Gets or sets the fraction of body parameters to adapt (for efficiency).
    /// </summary>
    /// <value>Default is 1.0 (adapt all body parameters).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your body has millions of parameters, adapting
    /// all of them might be slow. This setting lets you adapt only a fraction
    /// (e.g., 0.5 = adapt only half the body parameters).
    /// </para>
    /// </remarks>
    public double BodyAdaptationFraction { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use layer-wise learning rates for the body.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different layers might need different learning rates.
    /// Earlier layers (close to input) might need smaller updates than later layers.
    /// </para>
    /// </remarks>
    public bool UseLayerwiseLearningRates { get; set; } = false;

    /// <summary>
    /// Gets or sets the learning rate multiplier for earlier layers (if using layerwise rates).
    /// </summary>
    /// <value>Default is 0.1 (earlier layers update 10x slower).</value>
    public double EarlyLayerLrMultiplier { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the L2 regularization strength for the body.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double BodyL2Regularization { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to reinitialize the body for each task.
    /// </summary>
    /// <value>Default is false (use meta-learned initialization).</value>
    public bool ReinitializeBody { get; set; } = false;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the BOILOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The neural network to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public BOILOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all BOIL configuration options are properly set.
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
               NumMetaIterations > 0 &&
               BodyAdaptationFraction > 0 && BodyAdaptationFraction <= 1.0;
    }

    /// <summary>
    /// Creates a deep copy of the BOIL options.
    /// </summary>
    /// <returns>A new BOILOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new BOILOptions<T, TInput, TOutput>(MetaModel)
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
            BodyAdaptationFraction = BodyAdaptationFraction,
            UseLayerwiseLearningRates = UseLayerwiseLearningRates,
            EarlyLayerLrMultiplier = EarlyLayerLrMultiplier,
            BodyL2Regularization = BodyL2Regularization,
            ReinitializeBody = ReinitializeBody
        };
    }

    #endregion
}
