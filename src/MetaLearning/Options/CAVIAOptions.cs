using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Specifies how context parameters are injected into the model's computation.
/// </summary>
/// <remarks>
/// <para>
/// The injection mode determines how the task-specific context vector is combined
/// with the model's input or intermediate representations.
/// </para>
/// <para><b>For Beginners:</b> Think of the context as a "task description" that the
/// model uses to customize its behavior. The injection mode controls where and how
/// this description is fed into the model.
/// </para>
/// </remarks>
public enum CAVIAContextInjectionMode
{
    /// <summary>
    /// Concatenates the context vector with the input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The context vector is appended to the input, increasing the input dimension
    /// by the context dimension. This is the standard approach from the CAVIA paper.
    /// </para>
    /// <para><b>Use When:</b> You want the simplest and most general approach.
    /// Works well with any architecture.</para>
    /// </remarks>
    Concatenation,

    /// <summary>
    /// Adds the context vector element-wise to the input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Requires the context dimension to match the input dimension. The context
    /// acts as a learned bias shift for each input feature.
    /// </para>
    /// <para><b>Use When:</b> You want the context to modulate input features without
    /// changing the input dimension. Requires context dimension equals input dimension.</para>
    /// </remarks>
    Addition,

    /// <summary>
    /// Multiplies the context vector element-wise with the input features (FiLM-style gating).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Requires the context dimension to match the input dimension. The context
    /// acts as a learned scaling factor for each input feature.
    /// </para>
    /// <para><b>Use When:</b> You want the context to scale input features (e.g.,
    /// suppressing irrelevant features for a particular task).</para>
    /// </remarks>
    Multiplication
}

/// <summary>
/// Configuration options for the CAVIA (Fast Context Adaptation via Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// CAVIA separates model parameters into two groups:
/// 1. <b>Shared parameters (body):</b> Updated only in the outer loop across all tasks
/// 2. <b>Context parameters:</b> Task-specific, adapted in the inner loop per task
///
/// By only adapting context parameters in the inner loop, CAVIA is:
/// - Much faster than MAML (fewer parameters to differentiate through)
/// - Less prone to meta-overfitting (fewer adapted parameters = stronger regularization)
/// - Conceptually cleaner (explicit separation of shared vs. task-specific knowledge)
/// </para>
/// <para><b>For Beginners:</b> CAVIA is like MAML but smarter about what it adapts.
///
/// Imagine you're a chef learning to cook different cuisines:
/// - Your cooking skills (shared params) stay the same across cuisines
/// - But for each cuisine, you adjust your seasoning preferences (context params)
/// - CAVIA only adjusts the "seasoning" for each new task, not all your skills
/// - This makes adaptation much faster and less likely to overfit
///
/// The context vector is a small set of numbers that tells the model
/// "this is the kind of task we're dealing with." During adaptation,
/// only these numbers change - the main model stays fixed.
/// </para>
/// <para>
/// Reference: Zintgraf, L. M., Shiarli, K., Kurin, V., Hofmann, K., &amp; Whiteson, S. (2019).
/// Fast Context Adaptation via Meta-Learning. ICML 2019.
/// </para>
/// </remarks>
public class CAVIAOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model's parameters serve as the shared body parameters (phi).
    /// These are updated only during the outer loop of meta-training.
    /// </para>
    /// <para><b>For Beginners:</b> This is the neural network that processes inputs.
    /// CAVIA will learn good shared parameters for this model, plus a context vector
    /// that gets adapted for each new task.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses the model's default loss function).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The loss function measures how well the adapted model performs on query examples.
    /// Cross-entropy is typical for classification, MSE for regression.
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for outer loop updates (shared parameter updates).
    /// Default: null (uses built-in SGD with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates (context parameter updates).
    /// Default: null (uses built-in SGD with InnerLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (context parameter adaptation).
    /// </summary>
    /// <value>The inner learning rate. Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Controls how quickly context parameters are updated when adapting to a new task.
    /// Since only context parameters are adapted (much fewer than full model parameters),
    /// this can typically be set higher than MAML's inner learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This is how fast the model adjusts its "task description"
    /// (context) when given examples of a new task. A typical range is 0.001 to 0.1.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (shared parameter updates).
    /// </summary>
    /// <value>The outer learning rate. Default is 0.001.</value>
    /// <remarks>
    /// <para>
    /// Controls how quickly the shared body parameters are updated during meta-training.
    /// This is the learning rate for the "learning to learn" process.
    /// </para>
    /// <para><b>For Beginners:</b> This is how fast the model's core skills improve
    /// across all tasks. Start with 0.001 and adjust based on training stability.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps for inner loop context adaptation.
    /// </summary>
    /// <value>The number of adaptation steps. Default is 5.</value>
    /// <remarks>
    /// <para>
    /// CAVIA typically uses more inner steps than MAML because adapting only context
    /// parameters is computationally cheaper. 5-10 steps is common.
    /// </para>
    /// <para><b>For Beginners:</b> How many times to refine the context vector on
    /// each new task. More steps means better adaptation but slower. 5 is a good default.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>The meta-batch size. Default is 4.</value>
    /// <remarks>
    /// <para>
    /// Each training step samples this many tasks, adapts context for each,
    /// and averages the outer-loop gradients before updating shared parameters.
    /// </para>
    /// </remarks>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>The number of meta-iterations. Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>The gradient clip threshold, or null to disable. Default is 10.0.</value>
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
    /// Gets or sets whether to use first-order approximation for meta-gradients.
    /// </summary>
    /// <value>Default is true (recommended for CAVIA).</value>
    /// <remarks>
    /// <para>
    /// CAVIA's authors recommend first-order approximation because:
    /// 1. Context parameters are small, so second-order effects are minimal
    /// 2. First-order is much faster to compute
    /// 3. Empirically performs nearly as well as full second-order
    /// </para>
    /// <para><b>For Beginners:</b> Keep this as true. It makes training faster
    /// with negligible accuracy loss.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region CAVIA-Specific Properties

    /// <summary>
    /// Gets or sets the dimension of the context parameter vector.
    /// </summary>
    /// <value>The context dimension. Default is 100.</value>
    /// <remarks>
    /// <para>
    /// The context vector is a low-dimensional representation of task-specific information.
    /// It is initialized to zeros for each new task and adapted during the inner loop.
    /// Larger values allow more expressiveness but may lead to overfitting on small support sets.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much "task-specific memory" the model has.
    /// 100 is a good default. Use smaller values (e.g., 32-64) if tasks are very similar,
    /// or larger values (e.g., 128-256) if tasks are very different from each other.
    /// </para>
    /// </remarks>
    public int ContextDimension { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of context vectors to use.
    /// </summary>
    /// <value>The number of context vectors. Default is 1.</value>
    /// <remarks>
    /// <para>
    /// Multiple context vectors can be injected at different layers of the model.
    /// With 1 context vector, it is injected at the input level only.
    /// With N context vectors, they can be injected at N different points in the network.
    /// </para>
    /// <para><b>For Beginners:</b> Start with 1. Multiple context vectors are an advanced
    /// feature that can help when tasks differ at multiple levels of abstraction.
    /// </para>
    /// </remarks>
    public int NumContextVectors { get; set; } = 1;

    /// <summary>
    /// Gets or sets the initial value for context parameters at the start of each task.
    /// </summary>
    /// <value>The context initialization value. Default is 0.0.</value>
    /// <remarks>
    /// <para>
    /// The CAVIA paper initializes context parameters to zero before each task adaptation.
    /// This ensures a consistent starting point and helps the shared parameters learn
    /// to work well with zero context (the "default task").
    /// </para>
    /// <para><b>For Beginners:</b> Keep this at 0.0 as recommended by the paper.
    /// The model learns to use the context relative to this starting point.
    /// </para>
    /// </remarks>
    public double ContextInitValue { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets how the context vector is injected into the model's computation.
    /// </summary>
    /// <value>The context injection mode. Default is Concatenation.</value>
    /// <remarks>
    /// <para>
    /// Concatenation appends the context to the input, increasing the effective input
    /// dimension. Addition and Multiplication require the context dimension to match
    /// the input dimension and modulate features directly.
    /// </para>
    /// <para><b>For Beginners:</b> Use Concatenation (default) unless you have a specific
    /// reason to change it. Concatenation is the most flexible and is what the paper uses.
    /// </para>
    /// </remarks>
    public CAVIAContextInjectionMode ContextInjectionMode { get; set; } = CAVIAContextInjectionMode.Concatenation;

    /// <summary>
    /// Gets or sets whether to apply L2 regularization on context parameters during adaptation.
    /// </summary>
    /// <value>True to apply regularization; false otherwise. Default is false.</value>
    /// <remarks>
    /// <para>
    /// L2 regularization on context parameters prevents them from growing too large
    /// during inner-loop adaptation, which can help prevent overfitting on small
    /// support sets.
    /// </para>
    /// <para><b>For Beginners:</b> Enable this if you notice the model overfitting
    /// (performing well on support sets but poorly on query sets).
    /// </para>
    /// </remarks>
    public bool UseContextRegularization { get; set; } = false;

    /// <summary>
    /// Gets or sets the L2 regularization strength for context parameters.
    /// </summary>
    /// <value>The regularization coefficient. Default is 0.001.</value>
    /// <remarks>
    /// <para>
    /// Only used when <see cref="UseContextRegularization"/> is true.
    /// Higher values constrain context parameters closer to their initial values.
    /// </para>
    /// </remarks>
    public double ContextRegularizationStrength { get; set; } = 0.001;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the CAVIAOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The model to be meta-trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    /// <example>
    /// <code>
    /// // Create CAVIA with minimal configuration
    /// var options = new CAVIAOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myEncoder);
    /// var cavia = new CAVIAAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    ///
    /// // Create CAVIA with custom configuration
    /// var options = new CAVIAOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myEncoder)
    /// {
    ///     ContextDimension = 64,
    ///     AdaptationSteps = 10,
    ///     InnerLearningRate = 0.05,
    ///     ContextInjectionMode = CAVIAContextInjectionMode.Concatenation
    /// };
    /// </code>
    /// </example>
    public CAVIAOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all CAVIA configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Checks:
    /// - MetaModel is set
    /// - Learning rates are positive
    /// - Context dimension is positive
    /// - Batch sizes and iteration counts are positive
    /// - Regularization strength is non-negative (if enabled)
    /// </para>
    /// </remarks>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps > 0 &&
               ContextDimension > 0 &&
               NumContextVectors > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               EvaluationTasks > 0 &&
               (!UseContextRegularization || ContextRegularizationStrength >= 0);
    }

    /// <summary>
    /// Creates a deep copy of the CAVIA options.
    /// </summary>
    /// <returns>A new CAVIAOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new CAVIAOptions<T, TInput, TOutput>(MetaModel)
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
            ContextDimension = ContextDimension,
            NumContextVectors = NumContextVectors,
            ContextInitValue = ContextInitValue,
            ContextInjectionMode = ContextInjectionMode,
            UseContextRegularization = UseContextRegularization,
            ContextRegularizationStrength = ContextRegularizationStrength
        };
    }

    #endregion
}
