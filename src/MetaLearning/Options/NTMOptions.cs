using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Controller type for Neural Turing Machine.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The controller is the "brain" of the NTM that
/// decides what to read from and write to memory. LSTM is better for
/// sequential data, while MLP works for fixed-size inputs.
/// </para>
/// </remarks>
public enum NTMControllerType
{
    /// <summary>
    /// LSTM controller for sequential data processing.
    /// </summary>
    LSTM,

    /// <summary>
    /// MLP controller for fixed-size data processing.
    /// </summary>
    MLP
}

/// <summary>
/// Memory initialization strategies for NTM.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> How to fill the memory when starting.
/// Zeros is simplest, Random adds some noise, Learned uses trained patterns.
/// </para>
/// </remarks>
public enum NTMMemoryInitialization
{
    /// <summary>
    /// Initialize all memory values to zero.
    /// </summary>
    Zeros,

    /// <summary>
    /// Initialize with small random values.
    /// </summary>
    Random,

    /// <summary>
    /// Initialize with learned patterns.
    /// </summary>
    Learned
}

/// <summary>
/// Configuration options for Neural Turing Machine (NTM) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Neural Turing Machines augment neural networks with an external memory matrix
/// and differentiable attention mechanisms for reading and writing. This enables
/// algorithms to be learned and executed within the neural network itself.
/// </para>
/// <para><b>For Beginners:</b> NTM is like a neural computer with RAM:
///
/// 1. Controller network processes inputs like a CPU
/// 2. Generates read/write keys for memory access
/// 3. Attention mechanism determines where to read/write
/// 4. External memory stores information persistently
/// 5. Differentiable operations allow end-to-end learning
///
/// This allows learning algorithms like sorting, copying, and associative recall!
/// </para>
/// </remarks>
public class NTMOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (controller network) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model acts as the controller that processes inputs and generates
    /// read/write keys for memory operations.
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
    /// Default: null (NTM uses memory-based adaptation, not gradient-based inner loop).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (not used in NTM).
    /// </summary>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (controller training).
    /// </summary>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps.
    /// </summary>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
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
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region NTM-Specific Properties

    /// <summary>
    /// Gets or sets the type of controller to use.
    /// </summary>
    /// <value>Default is LSTM.</value>
    public NTMControllerType ControllerType { get; set; } = NTMControllerType.LSTM;

    /// <summary>
    /// Gets or sets the number of memory slots.
    /// </summary>
    /// <value>Default is 128.</value>
    public int MemorySize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension of each memory slot.
    /// </summary>
    /// <value>Default is 20.</value>
    public int MemoryWidth { get; set; } = 20;

    /// <summary>
    /// Gets or sets the number of read heads.
    /// </summary>
    /// <value>Default is 1.</value>
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
    /// Gets or sets the hidden size of the controller.
    /// </summary>
    /// <value>Default is 100.</value>
    public int ControllerHiddenSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to initialize memory at start of episodes.
    /// </summary>
    /// <value>Default is true.</value>
    public bool InitializeMemory { get; set; } = true;

    /// <summary>
    /// Gets or sets how to initialize memory.
    /// </summary>
    /// <value>Default is Zeros.</value>
    public NTMMemoryInitialization MemoryInitialization { get; set; } = NTMMemoryInitialization.Zeros;

    /// <summary>
    /// Gets or sets the memory usage regularization strength.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double MemoryUsageRegularization { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the memory sharpness regularization strength.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double MemorySharpnessRegularization { get; set; } = 0.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the NTMOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The controller network to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public NTMOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all NTM configuration options are properly set.
    /// </summary>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               MemorySize > 0 &&
               MemoryWidth > 0 &&
               NumReadHeads > 0 &&
               NumWriteHeads > 0 &&
               ControllerHiddenSize > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0;
    }

    /// <summary>
    /// Creates a deep copy of the NTM options.
    /// </summary>
    public IMetaLearnerOptions<T> Clone()
    {
        return new NTMOptions<T, TInput, TOutput>(MetaModel)
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
            ControllerType = ControllerType,
            MemorySize = MemorySize,
            MemoryWidth = MemoryWidth,
            NumReadHeads = NumReadHeads,
            NumWriteHeads = NumWriteHeads,
            NumClasses = NumClasses,
            ControllerHiddenSize = ControllerHiddenSize,
            InitializeMemory = InitializeMemory,
            MemoryInitialization = MemoryInitialization,
            MemoryUsageRegularization = MemoryUsageRegularization,
            MemorySharpnessRegularization = MemorySharpnessRegularization
        };
    }

    #endregion
}
