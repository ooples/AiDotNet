using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for MAML (Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// MAML learns an initialization that can be quickly fine-tuned to new tasks.
/// These options control both the inner loop (task adaptation) and outer loop (meta-optimization).
/// </para>
/// <para><b>For Beginners:</b> MAML has two learning loops:
/// - Inner loop: Fast adaptation to a specific task (uses InnerLearningRate, AdaptationSteps)
/// - Outer loop: Slow learning of good initialization (uses OuterLearningRate)
/// </para>
/// </remarks>
public class MAMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Default: null (uses model's default loss function if available).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for meta-parameter updates (outer loop).
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner-loop adaptation.
    /// Default: null (uses built-in SGD optimizer with InnerLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (task adaptation).
    /// </summary>
    /// <value>The inner learning rate. Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// In MAML, the inner learning rate controls how quickly the model adapts to a specific task
    /// during the K gradient steps of the inner loop. This rate is typically larger than the
    /// outer learning rate because adaptation needs to happen quickly with few examples.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how fast the model learns when it sees a new task.
    /// A higher value means faster but potentially less stable adaptation.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-optimization).
    /// </summary>
    /// <value>The outer learning rate. Default is 0.001.</value>
    /// <remarks>
    /// <para>
    /// The outer learning rate controls how the meta-parameters (initial weights) are updated
    /// after evaluating adaptation performance across a batch of tasks. This rate is typically
    /// 10x smaller than the inner rate because meta-learning requires careful, gradual updates.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how fast the model learns "how to learn."
    /// Too high can cause instability; too low means slow meta-training.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps to take during inner loop adaptation.
    /// </summary>
    /// <value>The number of adaptation steps. Default is 5.</value>
    /// <remarks>
    /// <para>
    /// This is the K in "K-shot learning." MAML performs K gradient descent steps on the
    /// support set of each task during the inner loop. The original MAML paper uses 1-10 steps.
    /// More steps allow better adaptation but increase computation and memory usage.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> How many times the model updates itself when learning a new task.
    /// 1 step = very fast but rough adaptation; 10 steps = slower but more refined.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>The meta-batch size. Default is 4.</value>
    /// <remarks>
    /// <para>
    /// Each meta-training step samples this many tasks, adapts to each one, and averages
    /// the meta-gradients. Larger batch sizes provide more stable gradient estimates but
    /// require more memory and computation per iteration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> How many different practice tasks to learn from before
    /// updating the model's "learning strategy."
    /// </para>
    /// </remarks>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations to perform.
    /// </summary>
    /// <value>The number of meta-iterations. Default is 1000.</value>
    /// <remarks>
    /// <para>
    /// The total number of outer loop updates. Each iteration processes MetaBatchSize tasks.
    /// The original MAML paper uses 60,000 iterations for Omniglot and 15,000 for MiniImageNet.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> How long to train the meta-learner. More iterations = better
    /// learning but longer training time.
    /// </para>
    /// </remarks>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation (FOMAML) instead of full MAML.
    /// </summary>
    /// <value>True to use FOMAML; false for full second-order MAML. Default is false.</value>
    /// <remarks>
    /// <para>
    /// First-order MAML (FOMAML) ignores the second-order derivatives that arise from
    /// differentiating through the inner loop adaptation. This dramatically reduces
    /// memory usage and computation time while maintaining competitive performance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A simpler, faster version of MAML that works almost as well.
    /// Set to true if you're running out of memory or training is too slow.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>The gradient clip threshold, or null to disable. Default is 10.0.</value>
    /// <remarks>
    /// <para>
    /// Gradient clipping prevents training instability by limiting the magnitude of gradients.
    /// In MAML, gradients can become large due to the nested optimization structure,
    /// making clipping especially important.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Prevents the model from taking steps that are too large,
    /// which could break training. Set to null to disable.
    /// </para>
    /// </remarks>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>The random seed, or null for non-deterministic behavior. Default is null.</value>
    /// <remarks>
    /// <para>
    /// Setting a seed ensures reproducible task sampling and weight initialization in MAML.
    /// This is useful for debugging and comparing experiments.
    /// </para>
    /// </remarks>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation during meta-training.
    /// </summary>
    /// <value>The number of evaluation tasks. Default is 100.</value>
    /// <remarks>
    /// <para>
    /// Periodically, MAML evaluates its generalization by adapting to and evaluating
    /// on this many held-out tasks. More tasks provide more reliable evaluation metrics.
    /// </para>
    /// </remarks>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often (in meta-iterations) to evaluate the meta-learner.
    /// </summary>
    /// <value>The evaluation frequency. Default is 100.</value>
    /// <remarks>
    /// <para>
    /// Every EvaluationFrequency meta-iterations, MAML will evaluate on EvaluationTasks
    /// tasks to track training progress. More frequent evaluation provides better
    /// visibility but slows down training.
    /// </para>
    /// </remarks>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save model checkpoints during meta-training.
    /// </summary>
    /// <value>True to enable checkpointing; false to disable. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, MAML will save the meta-parameters at regular intervals.
    /// This allows resuming training after interruption and keeping the best model.
    /// </para>
    /// </remarks>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often (in meta-iterations) to save checkpoints.
    /// </summary>
    /// <value>The checkpoint frequency. Default is 500.</value>
    /// <remarks>
    /// <para>
    /// When checkpointing is enabled, MAML saves the meta-parameters every
    /// CheckpointFrequency iterations. More frequent saves use more disk space
    /// but reduce potential data loss on interruption.
    /// </para>
    /// </remarks>
    public int CheckpointFrequency { get; set; } = 500;

    #endregion

    #region MAML-Specific Properties

    /// <summary>
    /// Gets or sets whether to use first-order approximation (FOMAML).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, ignores second-order derivatives during meta-gradient computation.
    /// This is faster and uses less memory, with minimal performance impact.
    /// </para>
    /// <para>Default: false (alias for UseFirstOrder property)</para>
    /// </remarks>
    public bool UseFirstOrderApproximation
    {
        get => UseFirstOrder;
        set => UseFirstOrder = value;
    }

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MAMLOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The meta-model to be trained (required).</param>
    public MAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all MAML configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid for MAML training; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks all critical hyperparameters required for MAML to function correctly:
    /// - MetaModel must be set (the neural network to meta-train)
    /// - Learning rates must be positive (both inner and outer loops)
    /// - Adaptation steps must be at least 1 (K gradient steps in inner loop)
    /// - Batch sizes and iteration counts must be positive
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Call this before training to ensure your settings make sense.
    /// If it returns false, check that all required values are set and positive.
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
               EvaluationTasks > 0;
    }

    /// <summary>
    /// Creates a deep copy of the MAML options.
    /// </summary>
    /// <returns>A new MAMLOptions instance with the same configuration values.</returns>
    /// <remarks>
    /// <para>
    /// Note: The MetaModel reference is shared (not deep-cloned) because models are
    /// typically expensive to copy and MAML manages its own model cloning internally
    /// during the meta-training process.
    /// </para>
    /// </remarks>
    public IMetaLearnerOptions<T> Clone()
    {
        return new MAMLOptions<T, TInput, TOutput>(MetaModel)
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
            CheckpointFrequency = CheckpointFrequency
        };
    }

    #endregion
}
