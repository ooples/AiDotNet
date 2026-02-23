using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Memory-Augmented Neural Networks (MANN) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Memory-Augmented Neural Networks combine a neural network controller with an external
/// memory matrix. The network can read from and write to this memory, enabling rapid
/// learning by storing new information directly in memory during adaptation.
/// </para>
/// <para><b>For Beginners:</b> MANN is like a neural network with a notebook:
///
/// 1. The "controller" (neural network) processes inputs
/// 2. The "memory" stores important information for later
/// 3. Read heads retrieve relevant memories
/// 4. Write heads store new information
///
/// This allows one-shot learning - see an example once, store it, use it later!
/// </para>
/// </remarks>
public class MANNOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (controller network) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model acts as the controller that processes inputs and generates
    /// keys for memory read/write operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is the neural network that makes decisions
    /// about what to read from and write to memory.
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
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (not used in MANN).
    /// </summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Note: MANN doesn't perform gradient-based inner loop adaptation.
    /// This value is kept for interface compatibility.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (controller training).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps.
    /// </summary>
    /// <value>Default is 1 (MANN uses memory-based adaptation).</value>
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
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default is true since MANN doesn't use gradient-based inner loop.</value>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region MANN-Specific Properties

    /// <summary>
    /// Gets or sets the number of memory slots.
    /// </summary>
    /// <value>Default is 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many "pages" in the memory notebook.
    /// More slots = more storage, but more computation.
    /// </para>
    /// </remarks>
    public int MemorySize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension of memory keys.
    /// </summary>
    /// <value>Default is 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the "index" used to find memories.
    /// Larger keys can encode more specific queries.
    /// </para>
    /// </remarks>
    public int MemoryKeySize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the dimension of memory values.
    /// </summary>
    /// <value>Default is 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much information each memory slot can hold.
    /// </para>
    /// </remarks>
    public int MemoryValueSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of read heads.
    /// </summary>
    /// <value>Default is 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different ways to query memory simultaneously.
    /// Multiple heads can retrieve different types of information.
    /// </para>
    /// </remarks>
    public int NumReadHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of write heads.
    /// </summary>
    /// <value>Default is 1.</value>
    public int NumWriteHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to clear memory between tasks.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, memory is partially cleared between tasks.
    /// If false, memory persists, enabling lifelong learning.
    /// </para>
    /// </remarks>
    public bool ClearMemoryBetweenTasks { get; set; } = false;

    /// <summary>
    /// Gets or sets the ratio of memory to retain when clearing.
    /// </summary>
    /// <value>Default is 0.5 (keep 50%).</value>
    public double MemoryRetentionRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use memory consolidation.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para>
    /// Memory consolidation periodically removes rarely-used memories
    /// to make room for new information.
    /// </para>
    /// </remarks>
    public bool UseMemoryConsolidation { get; set; } = false;

    /// <summary>
    /// Gets or sets the threshold for memory usage when consolidating.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double MemoryUsageThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use hierarchical memory organization.
    /// </summary>
    /// <value>Default is false.</value>
    public bool UseHierarchicalMemory { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to pre-initialize memory.
    /// </summary>
    /// <value>Default is false.</value>
    public bool UseMemoryPreInitialization { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to initialize with common patterns.
    /// </summary>
    /// <value>Default is false.</value>
    public bool UseCommonPatternsInitialization { get; set; } = false;

    /// <summary>
    /// Gets or sets the memory regularization strength.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double MemoryRegularization { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to project values before storing.
    /// </summary>
    /// <value>Default is false.</value>
    public bool UseValueProjection { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to apply softmax to output.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseOutputSoftmax { get; set; } = true;

    /// <summary>
    /// Gets or sets options for the read head configuration.
    /// </summary>
    public object? ReadHeadOptions { get; set; }

    /// <summary>
    /// Gets or sets options for the write head configuration.
    /// </summary>
    public object? WriteHeadOptions { get; set; }

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MANNOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The controller network to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    /// <example>
    /// <code>
    /// // Create MANN with minimal configuration
    /// var options = new MANNOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myController);
    /// var mann = new MANNAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    ///
    /// // Create MANN with custom memory configuration
    /// var options = new MANNOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myController)
    /// {
    ///     MemorySize = 256,
    ///     MemoryKeySize = 128,
    ///     NumReadHeads = 4
    /// };
    /// </code>
    /// </example>
    public MANNOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all MANN configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               MemorySize > 0 &&
               MemoryKeySize > 0 &&
               MemoryValueSize > 0 &&
               NumReadHeads > 0 &&
               NumWriteHeads > 0 &&
               NumClasses > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0;
    }

    /// <summary>
    /// Creates a deep copy of the MANN options.
    /// </summary>
    /// <returns>A new MANNOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new MANNOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
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
            MemorySize = MemorySize,
            MemoryKeySize = MemoryKeySize,
            MemoryValueSize = MemoryValueSize,
            NumReadHeads = NumReadHeads,
            NumWriteHeads = NumWriteHeads,
            NumClasses = NumClasses,
            ClearMemoryBetweenTasks = ClearMemoryBetweenTasks,
            MemoryRetentionRatio = MemoryRetentionRatio,
            UseMemoryConsolidation = UseMemoryConsolidation,
            MemoryUsageThreshold = MemoryUsageThreshold,
            UseHierarchicalMemory = UseHierarchicalMemory,
            UseMemoryPreInitialization = UseMemoryPreInitialization,
            UseCommonPatternsInitialization = UseCommonPatternsInitialization,
            MemoryRegularization = MemoryRegularization,
            UseValueProjection = UseValueProjection,
            UseOutputSoftmax = UseOutputSoftmax,
            ReadHeadOptions = ReadHeadOptions,
            WriteHeadOptions = WriteHeadOptions
        };
    }

    #endregion
}
