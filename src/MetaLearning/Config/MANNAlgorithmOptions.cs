using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for Memory-Augmented Neural Networks (MANN).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// MANN combines neural networks with external memory for rapid learning and
/// lifelong knowledge retention. This configuration controls the memory architecture
/// and read/write mechanisms.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how MANN uses its external memory:
///
/// Key parameters:
/// - <b>MemorySize:</b> Number of memory slots (capacity)
/// - <b>MemoryKeySize:</b> Dimension of memory keys (for addressing)
/// - <b>MemoryValueSize:</b> Dimension of memory values (stored information)
/// - <b>NumReadHeads:</b> How many read heads to use (parallel reads)
/// - <b>NumWriteHeads:</b> How many write heads to use (parallel writes)
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Hierarchical memory organization
/// - Memory consolidation and forgetting
/// - Sparse memory access for efficiency
/// - Multiple read/write heads
/// - Lifelong learning support
/// - Memory usage regularization
/// </para>
/// </remarks>
public class MANNAlgorithmOptions<T, TInput, TOutput> : MetaLearningOptions<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the neural network controller.
    /// </summary>
    /// <value>
    /// The controller network that processes inputs and generates memory keys.
    /// This is the main neural network component of MANN.
    /// </value>
    public INeuralNetwork<T>? Controller { get; set; }

    /// <summary>
    /// Gets or sets the size of the external memory.
    /// </summary>
    /// <value>
    /// Number of slots in the external memory matrix.
    /// Larger memory can store more information but uses more resources.
    /// Default is 128.
    /// </value>
    /// <remarks>
    /// Memory size guidelines:
    /// - 32-64: Small tasks, limited memory
    /// - 128-256: Standard few-shot scenarios
    /// - 512-1024: Complex tasks with many examples
    /// - 2048+: Lifelong learning scenarios
    /// </remarks>
    public int MemorySize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension of memory keys.
    /// </summary>
    /// <value>
    /// Size of the key vectors used for addressing memory.
    /// Keys determine where to read/write in memory.
    /// Default is 64.
    /// </value>
    public int MemoryKeySize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the dimension of memory values.
    /// </summary>
    /// <value>
    /// Size of the value vectors stored in memory.
    /// Values contain the actual information being stored.
    /// Default is 64.
    /// </value>
    public int MemoryValueSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of read heads.
    /// </summary>
    /// <value>
    /// Number of parallel read operations per query.
    /// Each head can focus on different aspects of memory.
    /// Default is 1.
    /// </value>
    /// <remarks>
    /// Multiple read heads allow the network to:
    /// - Retrieve diverse information simultaneously
    /// - Focus on different memory regions
    /// - Combine information from multiple sources
    /// </remarks>
    public int NumReadHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of write heads.
    /// </summary>
    /// <value>
    /// Number of parallel write operations per support example.
    /// Default is 1.
    /// </value>
    public int NumWriteHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to clear memory between tasks.
    /// </summary>
    /// <value>
    /// If true, clears or partially clears memory between different tasks.
    /// Helps with task isolation but loses cross-task knowledge.
    /// Default is true.
    /// </value>
    public bool ClearMemoryBetweenTasks { get; set; } = true;

    /// <summary>
    /// Gets or sets the memory retention ratio when clearing.
    /// </summary>
    /// <value>
    /// Fraction of memory to retain (0.0 to 1.0).
    /// Only used when ClearMemoryBetweenTasks is true.
    /// Default is 0.5 (keep 50%).
    /// </value>
    public double MemoryRetentionRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use memory consolidation.
    /// </summary>
    /// <value>
    /// If true, periodically consolidates and prunes memory.
    /// Helps maintain memory efficiency over long training.
    /// Default is false.
    /// </value>
    public bool UseMemoryConsolidation { get; set; } = false;

    /// <summary>
    /// Gets or sets the memory usage threshold for consolidation.
    /// </summary>
    /// <value>
    /// Threshold below which memory slots are considered rarely used.
    /// Default is 0.1.
    /// </value>
    public double MemoryUsageThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use hierarchical memory.
    /// </summary>
    /// <value>
    /// If true, organizes memory hierarchically (like cache levels).
    /// Can improve access efficiency for large memories.
    /// Default is false.
    /// </value>
    public bool UseHierarchicalMemory { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to pre-initialize memory.
    /// </summary>
    /// <value>
    /// If true, initializes memory with common patterns or random values.
    /// Can help with early training stability.
    /// Default is false.
    /// </value>
    public bool UseMemoryPreInitialization { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use common patterns for initialization.
    /// </summary>
    /// <value>
    /// If true, initializes memory with common patterns from similar tasks.
    /// Only used when UseMemoryPreInitialization is true.
    /// Default is false.
    /// </value>
    public bool UseCommonPatternsInitialization { get; set; } = false;

    /// <summary>
    /// Gets or sets the memory regularization coefficient.
    /// </summary>
    /// <value>
    /// Regularization strength for memory usage.
    /// Prevents over-reliance on memory storage.
    /// Default is 1e-4.
    /// </value>
    public double MemoryRegularization { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets whether to use value projection.
    /// </summary>
    /// <value>
    /// If true, applies learned projection to memory values.
    /// Can improve representation quality.
    /// Default is false.
    /// </value>
    public bool UseValueProjection { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use output softmax.
    /// </summary>
    /// <value>
    /// If true, applies softmax to output predictions.
    /// Default is true.
    /// </value>
    public bool UseOutputSoftmax { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of classes for classification.
    /// </summary>
    /// <value>
    /// Number of output classes.
    /// Determines the size of the output layer.
    /// Default is 5.
    /// </value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets the read head configuration.
    /// </summary>
    public object ReadHeadOptions { get; set; } = new object();

    /// <summary>
    /// Gets or sets the write head configuration.
    /// </summary>
    public object WriteHeadOptions { get; set; } = new object();

    /// <summary>
    /// Gets or sets the maximum memory age before forgetting.
    /// </summary>
    /// <value>
    /// Maximum age (in episodes) before memory slots are forgotten.
    /// 0 means no forgetting based on age.
    /// Default is 0.
    /// </value>
    public int MaxMemoryAge { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use sparse memory access.
    /// </summary>
    /// <value>
    /// If true, uses sparse attention for memory access.
    /// Reduces computation for large memories.
    /// Default is false.
    /// </value>
    public bool UseSparseMemoryAccess { get; set; } = false;

    /// <summary>
    /// Gets or sets the top-k value for sparse access.
    /// </summary>
    /// <value>
    /// Number of top memory slots to attend to.
    /// Only used when UseSparseMemoryAccess is true.
    /// Default is 10.
    /// </value>
    public int SparseTopK { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use memory interpolation.
    /// </summary>
    /// <value>
    /// If true, interpolates between nearest memory slots.
    /// Can improve memory efficiency.
    /// Default is false.
    /// </value>
    public bool UseMemoryInterpolation { get; set; } = false;

    /// <summary>
    /// Gets or sets the interpolation neighborhood size.
    /// </summary>
    /// <value>
    /// Number of nearest neighbors for interpolation.
    /// Only used when UseMemoryInterpolation is true.
    /// Default is 3.
    /// </value>
    public int InterpolationNeighborhoodSize { get; set; } = 3;

    /// <summary>
    /// Creates a default MANN configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on Neural Turing Machine paper:
    /// - Memory size: 128 slots
    /// - Key/Value dimension: 64
    /// - Single read/write head
    /// - Memory clearing between tasks
    /// </remarks>
    public MANNAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 1; // MANN doesn't need inner loop
    }

    /// <summary>
    /// Creates a MANN configuration with custom values.
    /// </summary>
    /// <param name="controller">The neural network controller.</param>
    /// <param name="memorySize">Size of external memory.</param>
    /// <param name="memoryKeySize">Dimension of memory keys.</param>
    /// <param name="memoryValueSize">Dimension of memory values.</param>
    /// <param name="numReadHeads">Number of read heads.</param>
    /// <param name="numWriteHeads">Number of write heads.</param>
    /// <param name="clearMemoryBetweenTasks">Whether to clear memory between tasks.</param>
    /// <param name="useMemoryConsolidation">Whether to use memory consolidation.</param>
    /// <param name="memoryRegularization">Memory regularization coefficient.</param>
    /// <param name="innerLearningRate">Learning rate for optimization.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public MANNAlgorithmOptions(
        INeuralNetwork<T> controller,
        int memorySize = 128,
        int memoryKeySize = 64,
        int memoryValueSize = 64,
        int numReadHeads = 1,
        int numWriteHeads = 1,
        bool clearMemoryBetweenTasks = true,
        bool useMemoryConsolidation = false,
        double memoryRegularization = 1e-4,
        double innerLearningRate = 0.001,
        int numEpisodes = 10000)
    {
        Controller = controller;
        MemorySize = memorySize;
        MemoryKeySize = memoryKeySize;
        MemoryValueSize = memoryValueSize;
        NumReadHeads = numReadHeads;
        NumWriteHeads = numWriteHeads;
        ClearMemoryBetweenTasks = clearMemoryBetweenTasks;
        UseMemoryConsolidation = useMemoryConsolidation;
        MemoryRegularization = memoryRegularization;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = 1; // MANN doesn't use inner loop
        NumEpisodes = numEpisodes;
    }

    /// <summary>
    /// Validates the configuration parameters.
    /// </summary>
    /// <returns>True if all parameters are valid, false otherwise.</returns>
    public override bool IsValid()
    {
        // Check base class validation
        if (!base.IsValid())
            return false;

        // Check controller
        if (Controller == null)
            return false;

        // Check memory parameters
        if (MemorySize <= 0 || MemorySize > 10000)
            return false;

        if (MemoryKeySize <= 0 || MemoryKeySize > 1024)
            return false;

        if (MemoryValueSize <= 0 || MemoryValueSize > 1024)
            return false;

        // Check read/write heads
        if (NumReadHeads <= 0 || NumReadHeads > 10)
            return false;

        if (NumWriteHeads <= 0 || NumWriteHeads > 10)
            return false;

        // Check retention ratio
        if (MemoryRetentionRatio < 0.0 || MemoryRetentionRatio > 1.0)
            return false;

        // Check usage threshold
        if (MemoryUsageThreshold < 0.0 || MemoryUsageThreshold > 1.0)
            return false;

        // Check regularization
        if (MemoryRegularization < 0.0 || MemoryRegularization > 1.0)
            return false;

        // Check number of classes
        if (NumClasses <= 1 || NumClasses > 1000)
            return false;

        // Check max memory age
        if (MaxMemoryAge < 0)
            return false;

        // Check sparse access parameters
        if (UseSparseMemoryAccess)
        {
            if (SparseTopK <= 0 || SparseTopK > MemorySize)
                return false;
        }

        // Check interpolation parameters
        if (UseMemoryInterpolation)
        {
            if (InterpolationNeighborhoodSize <= 0 ||
                InterpolationNeighborhoodSize > MemorySize)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets the effective memory size after considering sparsity.
    /// </summary>
    /// <returns>Effective memory size for computation.</returns>
    public int GetEffectiveMemorySize()
    {
        if (UseSparseMemoryAccess)
        {
            return Math.Min(SparseTopK, MemorySize);
        }
        return MemorySize;
    }

    /// <summary>
    /// Gets the total number of memory operations per query.
    /// </summary>
    /// <returns>Total read and write operations.</returns>
    public int GetTotalMemoryOperations()
    {
        return NumReadHeads + NumWriteHeads;
    }
}