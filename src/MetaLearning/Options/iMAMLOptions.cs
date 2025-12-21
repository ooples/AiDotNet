using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for iMAML (Implicit Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// iMAML extends MAML by using implicit differentiation to compute meta-gradients,
/// which allows for many more adaptation steps without increased memory usage.
/// </para>
/// <para>
/// <b>For Beginners:</b> iMAML is a more memory-efficient version of MAML.
/// Regular MAML needs to remember every adaptation step (expensive!).
/// iMAML uses a mathematical trick to get the same result with constant memory.
/// </para>
/// <para>
/// Key parameters specific to iMAML:
/// - LambdaRegularization: Controls stability of implicit gradient computation
/// - ConjugateGradientIterations: How many CG steps to solve the implicit equation
/// - ConjugateGradientTolerance: When to stop CG early if converged
/// </para>
/// </remarks>
public class iMAMLOptions<T, TInput, TOutput> : IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model must implement IFullModel to support parameter getting/setting
    /// and gradient computation required for iMAML's implicit differentiation.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// </summary>
    /// <value>Default: null (uses model's default loss function if available).</value>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for meta-parameter updates (outer loop).
    /// </summary>
    /// <value>Default: null (uses built-in Adam optimizer with OuterLearningRate).</value>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner-loop adaptation.
    /// </summary>
    /// <value>Default: null (uses built-in SGD optimizer with InnerLearningRate).</value>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// </summary>
    /// <value>Default: null (tasks must be provided manually to MetaTrain).</value>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (task adaptation).
    /// </summary>
    /// <value>Default: 0.01 (standard for meta-learning inner loops).</value>
    /// <remarks>
    /// <para>
    /// In iMAML, the inner learning rate controls how quickly the model adapts during
    /// the K gradient steps. Unlike MAML, iMAML can use many more steps without
    /// memory penalty, so this rate can be tuned more conservatively.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-optimization).
    /// </summary>
    /// <value>Default: 0.001 (typically 10x smaller than inner rate).</value>
    /// <remarks>
    /// <para>
    /// Controls how the meta-parameters are updated after computing implicit gradients.
    /// The implicit gradient computation in iMAML often provides more accurate gradients
    /// than FOMAML, so this rate can sometimes be set slightly higher.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps to take during inner loop adaptation.
    /// </summary>
    /// <value>Default: 10 (iMAML can use more steps than MAML due to constant memory cost).</value>
    /// <remarks>
    /// <para>
    /// One of iMAML's key advantages: because memory usage is constant regardless of
    /// adaptation steps, you can use 10, 20, or even more steps without running out
    /// of memory. More steps often lead to better task adaptation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Unlike MAML where more steps = more memory, iMAML can
    /// use many steps freely. Try 10-20 steps for better performance.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default: 4 (typical meta-batch size).</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations to perform.
    /// </summary>
    /// <value>Default: 1000 (typical meta-training length).</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default: false (use full implicit differentiation).</value>
    /// <remarks>
    /// <para>
    /// When true, falls back to a first-order approximation similar to FOMAML.
    /// This is faster but loses some of the benefits of implicit differentiation.
    /// Only set to true if you're having stability issues or need maximum speed.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default: 10.0 (prevents exploding gradients).</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>Default: null (non-deterministic).</value>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    /// <value>Default: 100.</value>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate during meta-training.
    /// </summary>
    /// <value>Default: 100 (evaluate every 100 iterations).</value>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save checkpoints during training.
    /// </summary>
    /// <value>Default: false.</value>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    /// <value>Default: 500.</value>
    public int CheckpointFrequency { get; set; } = 500;

    #endregion

    #region iMAML-Specific Properties

    /// <summary>
    /// Gets or sets the regularization strength for implicit gradient computation.
    /// </summary>
    /// <value>Default: 1.0 (balances stability and accuracy).</value>
    /// <remarks>
    /// <para>
    /// This parameter (often called lambda) controls the regularization in the
    /// implicit equation: (I + lambda * H)^(-1) * g, where H is the Hessian.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how "smooth" the implicit gradient is:
    /// - Higher values (e.g., 2.0, 5.0): More stable but less accurate gradients
    /// - Lower values (e.g., 0.5, 0.1): More accurate but potentially unstable
    /// - 1.0 is a good default that balances both concerns
    /// </para>
    /// </remarks>
    public double LambdaRegularization { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum number of Conjugate Gradient iterations.
    /// </summary>
    /// <value>Default: 10 (usually sufficient for convergence).</value>
    /// <remarks>
    /// <para>
    /// The Conjugate Gradient (CG) method is used to solve the linear system
    /// that arises in implicit differentiation. More iterations give more
    /// accurate solutions but take longer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how many steps to take when solving
    /// the implicit equation. 10-20 is usually enough.
    /// </para>
    /// </remarks>
    public int ConjugateGradientIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the convergence tolerance for Conjugate Gradient.
    /// </summary>
    /// <value>Default: 1e-10 (very precise convergence).</value>
    /// <remarks>
    /// <para>
    /// CG stops early if the residual falls below this threshold.
    /// Lower values are more precise but may require more iterations.
    /// </para>
    /// </remarks>
    public double ConjugateGradientTolerance { get; set; } = 1e-10;

    /// <summary>
    /// Gets or sets whether to use the Neumann series approximation for implicit gradients.
    /// </summary>
    /// <value>Default: false (use full CG solver).</value>
    /// <remarks>
    /// <para>
    /// The Neumann series provides a faster but approximate solution to the
    /// implicit equation. Set to true for faster training at the cost of
    /// some gradient accuracy.
    /// </para>
    /// </remarks>
    public bool UseNeumannApproximation { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of terms in the Neumann series approximation.
    /// </summary>
    /// <value>Default: 5 (sufficient for most cases).</value>
    /// <remarks>
    /// <para>
    /// Only used when UseNeumannApproximation is true. More terms give
    /// better approximations but take longer to compute.
    /// </para>
    /// </remarks>
    public int NeumannSeriesTerms { get; set; } = 5;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the iMAMLOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The meta-model to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public iMAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all iMAML configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid for iMAML training; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Checks all required hyperparameters including iMAML-specific ones:
    /// - Standard meta-learning parameters (learning rates, steps, etc.)
    /// - Lambda regularization must be non-negative
    /// - CG parameters must be positive
    /// </para>
    /// </remarks>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               EvaluationTasks > 0 &&
               LambdaRegularization >= 0 &&
               ConjugateGradientIterations > 0 &&
               ConjugateGradientTolerance > 0 &&
               ConjugateGradientTolerance < 1.0 &&
               NeumannSeriesTerms > 0;
    }

    /// <summary>
    /// Creates a deep copy of the iMAML options.
    /// </summary>
    /// <returns>A new iMAMLOptions instance with the same configuration values.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new iMAMLOptions<T, TInput, TOutput>(MetaModel)
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
            UseFirstOrder = UseFirstOrder,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            LambdaRegularization = LambdaRegularization,
            ConjugateGradientIterations = ConjugateGradientIterations,
            ConjugateGradientTolerance = ConjugateGradientTolerance,
            UseNeumannApproximation = UseNeumannApproximation,
            NeumannSeriesTerms = NeumannSeriesTerms
        };
    }

    #endregion
}
