using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

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
public class BOILOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    public int? RandomSeed { get => Seed; set => Seed = value; }

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
    /// Gets or sets whether to drop the second-order terms of the meta-gradient.
    /// </summary>
    /// <value>Default is false: the exact meta-gradient, as MAML's outer loop computes it.</value>
    /// <remarks>
    /// BOIL's outer loop "updates the meta-initialized parameters using the meta-loss" as MAML does (Oh et al. 2021,
    /// Sec. 2.1). The exact gradient runs through every body step: body Hessian-vector products, and the head's cross
    /// term through each step's support gradient. True keeps only the query gradient at the adapted body.
    /// </remarks>
    public bool UseFirstOrder { get; set; } = false;

    #endregion

    #region BOIL-Specific Properties

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets the width of each example's representation the head reads: the body's per-example output width.
    /// </summary>
    /// <value>Default is 512.</value>
    public int FeatureDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the fraction of the body's layers the inner loop adapts, counted from the top.
    /// </summary>
    /// <value>Default is 1.0 (adapt every body layer, the paper's setting).</value>
    /// <remarks>
    /// <para>
    /// An extension. BOIL's analysis finds representation change concentrated in the high-level body and reuse in
    /// the low and middle layers (Oh et al. 2021, Sec. 4), so adapting only the top layers keeps the change where it
    /// happens and costs less. Layers come from the body's own layer structure; a body with none counts as one layer.
    /// At least one layer is always adapted.
    /// </para>
    /// </remarks>
    public double BodyAdaptationFraction { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to meta-learn one inner learning rate per body layer and per inner step.
    /// </summary>
    /// <value>Default is false (one shared inner learning rate, the paper's setting).</value>
    /// <remarks>
    /// <para>
    /// MAML++'s LSLR (Antoniou et al. 2019): "a learning rate and direction for each layer in the network as well as
    /// ... different learning rates for each adaptation" step. The rates are trained by the outer loop on the exact
    /// meta-gradient; their sign can flip a layer's update.
    /// </para>
    /// </remarks>
    public bool UseLayerwiseLearningRates { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial scale of the learned rates in the lower half of the body's layers.
    /// </summary>
    /// <value>Default is 0.1 (lower layers start 10x slower; the outer loop then learns every rate).</value>
    public double EarlyLayerLrMultiplier { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the L2 regularization strength for the body.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double BodyL2Regularization { get; set; } = 0.0;


    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the BOILOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The neural network to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public BOILOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
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
            BodyL2Regularization = BodyL2Regularization
        };
    }

    #endregion
}
