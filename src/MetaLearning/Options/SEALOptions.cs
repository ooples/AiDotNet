using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// SEAL is a gradient-based meta-learning algorithm that combines ideas from MAML with
/// sample-efficiency improvements. It learns initial parameters that can be quickly
/// adapted to new tasks with just a few examples, incorporating temperature scaling,
/// entropy regularization, and optional adaptive learning rates.
/// </para>
/// <para>
/// <b>For Beginners:</b> SEAL learns the best starting point for a model so that
/// it can quickly adapt to new tasks with minimal data. Think of it like learning
/// how to learn - after seeing many tasks, the model knows how to pick up new skills
/// quickly.
///
/// Imagine learning to play musical instruments:
/// - Learning your first instrument (piano) is hard
/// - Learning your second instrument (guitar) is easier
/// - By your 5th instrument, you've learned principles that help you
///   pick up any new instrument much faster
///
/// SEAL does the same with machine learning models!
/// </para>
/// <para>
/// Key features of SEAL:
/// - Temperature scaling: Controls confidence in predictions during meta-training
/// - Entropy regularization: Encourages diverse predictions to prevent overconfident models
/// - Adaptive learning rates: Per-parameter learning rate adaptation based on gradient norms
/// - Weight decay: Prevents overfitting to meta-training tasks
/// </para>
/// </remarks>
public class SEALOptions<T, TInput, TOutput> : IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model must implement IFullModel to support parameter getting/setting
    /// and gradient computation required for SEAL's meta-learning process.
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
    /// <remarks>
    /// <para>
    /// The meta-optimizer updates the initial model parameters based on performance
    /// across all tasks in the meta-batch. Adam is recommended for stable convergence.
    /// </para>
    /// </remarks>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner-loop adaptation.
    /// </summary>
    /// <value>Default: null (uses built-in SGD optimizer with InnerLearningRate).</value>
    /// <remarks>
    /// <para>
    /// The inner optimizer performs task-specific adaptation on the support set.
    /// SGD is typically used for simplicity and computational efficiency.
    /// </para>
    /// </remarks>
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
    /// This controls how quickly the model adapts to each task during the inner loop.
    /// A smaller value leads to more stable but slower adaptation; larger values
    /// can cause unstable training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like how big of steps you take when learning
    /// a specific task. Smaller steps are safer but slower; bigger steps are faster
    /// but might overshoot the optimal solution.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-optimization).
    /// </summary>
    /// <value>Default: 0.001 (typically 10x smaller than inner rate).</value>
    /// <remarks>
    /// <para>
    /// This controls how the meta-parameters (initial model weights) are updated
    /// based on performance across all tasks. Typically smaller than inner learning
    /// rate for stable meta-learning.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps to take during inner loop adaptation.
    /// </summary>
    /// <value>Default: 5 (standard for few-shot learning scenarios).</value>
    /// <remarks>
    /// <para>
    /// More adaptation steps allow better task-specific fitting but increase
    /// computational cost and memory usage (especially for second-order methods).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how many practice rounds you get on each task
    /// before being tested. More practice usually helps, but takes more time.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default: 4 (typical meta-batch size).</value>
    /// <remarks>
    /// <para>
    /// Larger batch sizes provide more stable gradient estimates but require
    /// more memory and computation per iteration.
    /// </para>
    /// </remarks>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations to perform.
    /// </summary>
    /// <value>Default: 1000 (typical meta-training length).</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation (FOMAML-style).
    /// </summary>
    /// <value>Default: true (recommended for computational efficiency).</value>
    /// <remarks>
    /// <para>
    /// First-order approximation ignores second-order gradient terms, reducing
    /// computational complexity from O(n^3) to O(n) with minimal performance loss.
    /// Most production implementations use first-order methods.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When true, SEAL uses a faster but slightly less accurate
    /// way to compute gradients. In practice, this works almost as well as the
    /// exact method but is much faster.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default: 10.0 (prevents exploding gradients).</value>
    /// <remarks>
    /// <para>
    /// Gradient clipping limits the magnitude of gradients during training,
    /// preventing numerical instability from exploding gradients.
    /// </para>
    /// </remarks>
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

    #region SEAL-Specific Properties

    /// <summary>
    /// Gets or sets whether to use adaptive inner learning rates.
    /// </summary>
    /// <value>Default: false (use fixed inner learning rate).</value>
    /// <remarks>
    /// <para>
    /// When enabled, SEAL computes per-parameter learning rates based on gradient
    /// norms during inner loop adaptation. Parameters with larger gradients get
    /// smaller learning rates (similar to AdaGrad's approach).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Instead of using the same step size for all parameters,
    /// adaptive learning rates adjust the step size for each parameter based on
    /// how much it has been changing. Parameters that change a lot get smaller steps
    /// to prevent overshooting.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveInnerLR { get; set; } = false;

    /// <summary>
    /// Gets or sets the temperature scaling factor for the loss function.
    /// </summary>
    /// <value>Default: 1.0 (no temperature scaling).</value>
    /// <remarks>
    /// <para>
    /// Temperature scaling divides the loss by the temperature value:
    /// - Temperature > 1.0: Softens predictions, reduces confidence
    /// - Temperature &lt; 1.0: Sharpens predictions, increases confidence
    /// - Temperature = 1.0: No effect (standard loss computation)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Temperature is like adjusting how "certain" the model
    /// should be about its predictions:
    /// - High temperature: Model becomes more humble, spreads probability across options
    /// - Low temperature: Model becomes more confident, concentrates probability on top choice
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the entropy regularization coefficient.
    /// </summary>
    /// <value>Default: 0.0 (no entropy regularization).</value>
    /// <remarks>
    /// <para>
    /// Entropy regularization adds a bonus for diverse predictions:
    /// Loss = Original_Loss - EntropyCoefficient * Entropy(predictions)
    ///
    /// Higher values encourage the model to be less confident and more exploratory,
    /// which can help prevent overfitting on meta-training tasks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Entropy measures how "spread out" the model's predictions are.
    /// By adding entropy to the objective, we encourage the model to not be too confident,
    /// which helps it generalize better to new tasks.
    /// </para>
    /// </remarks>
    public double EntropyCoefficient { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the weight decay (L2 regularization) coefficient.
    /// </summary>
    /// <value>Default: 0.0 (no weight decay).</value>
    /// <remarks>
    /// <para>
    /// Weight decay adds a penalty proportional to the squared magnitude of weights:
    /// Gradient = Original_Gradient + WeightDecay * Parameters
    ///
    /// This helps prevent overfitting by keeping weights small.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Weight decay is like a "simplicity penalty" that
    /// discourages the model from having very large weights. Large weights often
    /// indicate overfitting, so keeping them small helps the model generalize.
    /// </para>
    /// </remarks>
    public double WeightDecay { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the mode for adaptive learning rate computation.
    /// </summary>
    /// <value>Default: GradientNorm (scale by inverse gradient norm).</value>
    /// <remarks>
    /// <para>
    /// Different modes for computing adaptive learning rates:
    /// - GradientNorm: lr = base_lr / (sqrt(grad^2) + epsilon) [AdaGrad-like]
    /// - RunningMean: Maintains exponential moving average of squared gradients [RMSprop-like]
    /// - PerLayer: Applies same adaptive rate to all parameters in each layer
    /// </para>
    /// </remarks>
    public SEALAdaptiveLearningRateMode AdaptiveLearningRateMode { get; set; } = SEALAdaptiveLearningRateMode.GradientNorm;

    /// <summary>
    /// Gets or sets the epsilon value for numerical stability in adaptive learning rates.
    /// </summary>
    /// <value>Default: 1e-8 (small value to prevent division by zero).</value>
    /// <remarks>
    /// <para>
    /// Used in the denominator when computing adaptive learning rates to prevent
    /// division by zero: lr = base_lr / (sqrt(squared_grad) + epsilon)
    /// </para>
    /// </remarks>
    public double AdaptiveLearningRateEpsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the decay rate for running mean in adaptive learning rates.
    /// </summary>
    /// <value>Default: 0.99 (slow decay for stable estimates).</value>
    /// <remarks>
    /// <para>
    /// Only used when AdaptiveLearningRateMode is RunningMean.
    /// Higher values give more weight to historical gradients (more stable but slower to adapt).
    /// </para>
    /// </remarks>
    public double AdaptiveLearningRateDecay { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets whether to apply entropy regularization only during meta-training.
    /// </summary>
    /// <value>Default: true (entropy regularization disabled during adaptation).</value>
    /// <remarks>
    /// <para>
    /// When true, entropy regularization is only applied during meta-training
    /// (outer loop) and not during task adaptation (inner loop). This is often
    /// desirable as we want focused adaptation during the inner loop.
    /// </para>
    /// </remarks>
    public bool EntropyOnlyDuringMetaTrain { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum temperature for temperature annealing.
    /// </summary>
    /// <value>Default: 1.0 (no annealing, constant temperature).</value>
    /// <remarks>
    /// <para>
    /// When MinTemperature is less than Temperature, the temperature will linearly
    /// decrease from Temperature to MinTemperature over the course of training.
    /// This allows the model to be more exploratory early and more confident later.
    /// </para>
    /// </remarks>
    public double MinTemperature { get; set; } = 1.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the SEALOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The meta-model to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    /// <example>
    /// <code>
    /// // Create SEAL options with minimal configuration (uses all defaults)
    /// var options = new SEALOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(myNeuralNetwork);
    /// var seal = new SEALAlgorithm&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(options);
    ///
    /// // Create SEAL with entropy regularization for better generalization
    /// var options = new SEALOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(myNeuralNetwork)
    /// {
    ///     EntropyCoefficient = 0.01,
    ///     Temperature = 1.5,
    ///     UseAdaptiveInnerLR = true
    /// };
    ///
    /// // Create SEAL with weight decay and gradient clipping
    /// var options = new SEALOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(myNeuralNetwork)
    /// {
    ///     WeightDecay = 0.001,
    ///     GradientClipThreshold = 5.0,
    ///     AdaptationSteps = 10
    /// };
    /// </code>
    /// </example>
    public SEALOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all SEAL configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid for SEAL training; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Checks all required hyperparameters including SEAL-specific ones:
    /// - Standard meta-learning parameters (learning rates, steps, etc.)
    /// - Temperature must be positive
    /// - Entropy coefficient must be non-negative
    /// - Weight decay must be non-negative
    /// - Adaptive learning rate parameters must be valid
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
               Temperature > 0 &&
               EntropyCoefficient >= 0 &&
               WeightDecay >= 0 &&
               MinTemperature > 0 &&
               MinTemperature <= Temperature &&
               AdaptiveLearningRateEpsilon > 0 &&
               AdaptiveLearningRateDecay > 0 && AdaptiveLearningRateDecay <= 1.0;
    }

    /// <summary>
    /// Creates a deep copy of the SEAL options.
    /// </summary>
    /// <returns>A new SEALOptions instance with the same configuration values.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new SEALOptions<T, TInput, TOutput>(MetaModel)
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
            UseAdaptiveInnerLR = UseAdaptiveInnerLR,
            Temperature = Temperature,
            EntropyCoefficient = EntropyCoefficient,
            WeightDecay = WeightDecay,
            AdaptiveLearningRateMode = AdaptiveLearningRateMode,
            AdaptiveLearningRateEpsilon = AdaptiveLearningRateEpsilon,
            AdaptiveLearningRateDecay = AdaptiveLearningRateDecay,
            EntropyOnlyDuringMetaTrain = EntropyOnlyDuringMetaTrain,
            MinTemperature = MinTemperature
        };
    }

    #endregion
}

/// <summary>
/// Specifies the mode for computing adaptive learning rates in SEAL.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Different ways to adjust the step size for each parameter:
/// - GradientNorm: Big gradients get smaller steps (like AdaGrad)
/// - RunningMean: Uses a moving average of past gradients (like RMSprop)
/// - PerLayer: All parameters in a layer share the same adaptive rate
/// </para>
/// </remarks>
public enum SEALAdaptiveLearningRateMode
{
    /// <summary>
    /// Computes adaptive learning rate based on instantaneous gradient norm.
    /// lr = base_lr / (sqrt(grad^2) + epsilon)
    /// Similar to AdaGrad's approach.
    /// </summary>
    GradientNorm,

    /// <summary>
    /// Uses exponential moving average of squared gradients.
    /// Similar to RMSprop's approach, providing more stable estimates.
    /// </summary>
    RunningMean,

    /// <summary>
    /// Computes one adaptive rate per layer (averaged across layer parameters).
    /// Reduces noise in adaptive rate estimates for small layers.
    /// </summary>
    PerLayer
}
