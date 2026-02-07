using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a simple first-order meta-learning algorithm that doesn't require computing
/// gradients through the adaptation process. Instead, it interpolates between current
/// meta-parameters and adapted parameters.
/// </para>
/// <para>
/// <b>For Beginners:</b> Reptile is the simplest meta-learning algorithm to understand:
/// 1. Train on a task for several steps
/// 2. Move the starting point slightly toward where you ended up
/// 3. Repeat with many tasks
///
/// After seeing many tasks, your starting point becomes great for learning any new task!
/// </para>
/// <para>
/// Key difference from MAML: Reptile doesn't compute gradients through adaptation.
/// This makes it much simpler to implement and faster to run, with competitive performance.
/// </para>
/// </remarks>
public class ReptileOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
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
    /// <value>Default: null (Reptile uses simple interpolation, not optimizer-based updates).</value>
    /// <remarks>
    /// <para>
    /// Note: Reptile typically doesn't use an optimizer for outer-loop updates.
    /// Instead, it uses a simple interpolation step. This is provided for compatibility
    /// with the base interface but is not used in standard Reptile.
    /// </para>
    /// </remarks>
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
    /// <value>Default: 0.01.</value>
    /// <remarks>
    /// <para>
    /// This controls how fast the model learns during adaptation on each task.
    /// Reptile typically uses standard SGD with this learning rate.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-update).
    /// </summary>
    /// <value>Default: 1.0 (combined with Interpolation to control update magnitude).</value>
    /// <remarks>
    /// <para>
    /// In Reptile, this is multiplied with the Interpolation factor to control
    /// how far to move toward the adapted parameters. The effective step size
    /// is OuterLearningRate * Interpolation.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of gradient steps to take during inner loop adaptation.
    /// </summary>
    /// <value>Default: 5.</value>
    /// <remarks>
    /// <para>
    /// This is multiplied by InnerBatches to get the total number of inner steps.
    /// Total steps = AdaptationSteps * InnerBatches
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default: 5 (Reptile typically uses smaller batches than MAML).</value>
    public int MetaBatchSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations to perform.
    /// </summary>
    /// <value>Default: 10000 (Reptile often requires more iterations than MAML).</value>
    public int NumMetaIterations { get; set; } = 10000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default: true (Reptile is inherently first-order).</value>
    /// <remarks>
    /// <para>
    /// Reptile is inherently a first-order algorithm - it never computes gradients
    /// through the adaptation process. This property is always effectively true.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default: 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>Default: null (non-deterministic).</value>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    /// <value>Default: 100.</value>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate during meta-training.
    /// </summary>
    /// <value>Default: 500.</value>
    public int EvaluationFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to save checkpoints during training.
    /// </summary>
    /// <value>Default: false.</value>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    /// <value>Default: 1000.</value>
    public int CheckpointFrequency { get; set; } = 1000;

    #endregion

    #region Reptile-Specific Properties

    /// <summary>
    /// Gets or sets the interpolation factor for meta-updates.
    /// </summary>
    /// <value>Default: 1.0.</value>
    /// <remarks>
    /// <para>
    /// This controls what fraction of the adaptation direction to use.
    /// The meta-update is: theta_new = theta_old + (OuterLearningRate * Interpolation) * (theta_adapted - theta_old)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how far to move toward the adapted parameters:
    /// - 1.0: Move the full step (scaled by outer learning rate)
    /// - 0.5: Move only halfway
    /// - 0.1: Move just a little bit
    ///
    /// Smaller values are more conservative but slower to learn.
    /// </para>
    /// </remarks>
    public double Interpolation { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of inner batches per adaptation step.
    /// </summary>
    /// <value>Default: 1.</value>
    /// <remarks>
    /// <para>
    /// Total inner loop steps = AdaptationSteps * InnerBatches.
    /// This allows fine-grained control over adaptation depth.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Reptile can use many inner steps since it doesn't
    /// need to backprop through them. More steps = better task adaptation,
    /// which can help with harder tasks.
    /// </para>
    /// </remarks>
    public int InnerBatches { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to use the serial (single-task) or batched update variant.
    /// </summary>
    /// <value>Default: false (use batched updates for stability).</value>
    /// <remarks>
    /// <para>
    /// Serial Reptile: Updates meta-parameters after each single task (more variance)
    /// Batched Reptile: Averages updates across a batch of tasks (more stable)
    ///
    /// The batched variant is recommended for most cases.
    /// </para>
    /// </remarks>
    public bool UseSerialUpdate { get; set; } = false;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the ReptileOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The meta-model to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public ReptileOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all Reptile configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid for Reptile training; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               EvaluationTasks > 0 &&
               Interpolation > 0 && Interpolation <= 1.0 &&
               InnerBatches > 0;
    }

    /// <summary>
    /// Creates a deep copy of the Reptile options.
    /// </summary>
    /// <returns>A new ReptileOptions instance with the same configuration values.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new ReptileOptions<T, TInput, TOutput>(MetaModel)
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
            Interpolation = Interpolation,
            InnerBatches = InnerBatches,
            UseSerialUpdate = UseSerialUpdate
        };
    }

    #endregion
}
