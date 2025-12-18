using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for Neural Turing Machine (NTM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Neural Turing Machine combines neural networks with external memory that can
/// be read from and written to using differentiable attention mechanisms. This configuration
/// controls the memory architecture and controller design.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how the NTM's memory system works:
///
/// Key parameters:
/// - <b>ControllerType:</b> LSTM for sequences, MLP for fixed-size inputs
/// - <b>MemorySize:</b> Number of memory slots (like RAM capacity)
/// - <b>MemoryWidth:</b> Dimension of each memory slot (word size)
/// - <b>NumReadHeads:</b> Parallel read operations (multi-tasking)
/// - <b>AddressingMode:</b> How to access memory (content, location, mixed)
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Content and location-based addressing
/// - Interpolative memory writing
/// - Memory initialization strategies
/// - Multiple read/write heads
/// - Controller architectures (LSTM, MLP, GRU)
/// - Memory regularization for stability
/// </para>
/// </remarks>
public class NTMAlgorithmOptions<T, TInput, TOutput> : MetaLearningOptions<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the neural network controller.
    /// </summary>
    /// <value>
    /// The controller network that processes inputs and controls memory access.
    /// Can be LSTM, MLP, or other architectures.
    /// </value>
    public INeuralNetwork<T>? Controller { get; set; }

    /// <summary>
    /// Gets or sets the type of controller to use.
    /// </summary>
    /// <value>
    /// The architecture type for the controller.
    /// Default is LSTM.
    /// </value>
    /// <remarks>
    /// <b>Controller Types:</b>
    /// - <b>LSTM:</b> Best for sequential data, has internal memory
    /// - <b>MLP:</b> Best for fixed-size data, simpler architecture
    /// - <b>GRU:</b> Simpler than LSTM, good for medium sequences
    /// - <b>Transformer:</b> Best for long sequences with attention
    /// </remarks>
    public ControllerType ControllerType { get; set; } = ControllerType.LSTM;

    /// <summary>
    /// Gets or sets the size of the external memory.
    /// </summary>
    /// <value>
    /// Number of memory locations (like RAM capacity).
    /// Larger memory allows storing more information.
    /// Default is 128.
    /// </value>
    /// <remarks>
    /// Memory size guidelines:
    /// - 32-64: Small tasks, minimal storage needs
    /// - 128-256: Standard algorithmic tasks
    /// - 512-1024: Complex reasoning tasks
    /// - 2048+: Very large algorithms or datasets
    /// </remarks>
    public int MemorySize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the width of each memory location.
    /// </summary>
    /// <value>
    /// Dimension of vectors stored in memory.
    /// Determines information capacity per slot.
    /// Default is 64.
    /// </value>
    /// <remarks>
    /// Memory width affects:
    /// - Information density per memory location
    /// - Controller output dimension
    /// - Computational complexity
    /// </remarks>
    public int MemoryWidth { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of read heads.
    /// </summary>
    /// <value>
    /// Number of parallel read operations.
    /// Each head can focus on different information.
    /// Default is 1.
    /// </value>
    public int NumReadHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of write heads.
    /// </summary>
    /// <value>
    /// Number of parallel write operations.
    /// Most NTM implementations use a single write head.
    /// Default is 1.
    /// </value>
    public int NumWriteHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the addressing mode for memory access.
    /// </summary>
    /// <value>
    /// How to determine memory locations to read/write.
    /// Default is ContentBased.
    /// </value>
    /// <remarks>
    /// <b>Addressing Modes:</b>
    /// - <b>ContentBased:</b> Use similarity to stored content (most common)
    /// - <b>LocationBased:</b> Use absolute positions (like registers)
    /// - <b>Mixed:</b> Combine content and location addressing
    /// </remarks>
    public AddressingMode AddressingMode { get; set; } = AddressingMode.ContentBased;

    /// <summary>
    /// Gets or sets whether to initialize memory.
    /// </summary>
    /// <value>
    /// If true, initializes memory with default values.
    /// If false, starts with empty memory.
    /// Default is true.
    /// </value>
    public bool InitializeMemory { get; set; } = true;

    /// <summary>
    /// Gets or sets the memory initialization strategy.
    /// </summary>
    /// <value>
    /// How to initialize memory when InitializeMemory is true.
    /// Default is Zeros.
    /// </value>
    public MemoryInitialization MemoryInitialization { get; set; } = MemoryInitialization.Zeros;

    /// <summary>
    /// Gets or sets the memory usage regularization coefficient.
    /// </summary>
    /// <value>
    /// Regularization strength for memory usage.
    /// Prevents over-reliance on external memory.
    /// Default is 1e-5.
    /// </value>
    public double MemoryUsageRegularization { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the memory sharpness regularization coefficient.
    /// </summary>
    /// <value>
    /// Regularization to prevent overly sharp attention.
    /// Encourages smoother memory access patterns.
    /// Default is 1e-5.
    /// </value>
    public double MemorySharpnessRegularization { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use interpolation for memory writing.
    /// </summary>
    /// <value>
    /// If true, uses interpolation between old and new values.
    /// Allows smoother memory updates.
    /// Default is true.
    /// </value>
    public bool UseInterpolation { get; set; } = true;

    /// <summary>
    /// Gets or sets the interpolation strength.
    /// </summary>
    /// <value>
    /// Strength of interpolation (0.0 to 1.0).
    /// Controls how much old value is retained.
    /// Default is 1.0 (full interpolation).
    /// </value>
    public double InterpolationStrength { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the shift range for location-based addressing.
    /// </summary>
    /// <value>
    /// Maximum shift from previous location.
    /// Only used with LocationBased addressing.
    /// Default is 1.
    /// </value>
    public int ShiftRange { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to use temporal linking.
    /// </summary>
    /// <value>
    /// If true, links memory writes temporally.
    /// Helps with sequential tasks.
    /// Default is false.
    /// </value>
    public bool UseTemporalLinkage { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of controller hidden units.
    /// </summary>
    /// <value>
    /// Size of hidden layers in the controller.
    /// Affects network capacity.
    /// Default is 100.
    /// </value>
    public int ControllerHiddenUnits { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of controller layers.
    /// </summary>
    /// <value>
    /// Number of layers in the controller network.
    /// Deeper networks can learn more complex operations.
    /// Default is 1.
    /// </value>
    public int ControllerNumLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>
    /// If true, applies layer normalization in controller.
    /// Helps with training stability.
    /// Default is true.
    /// </value>
    public bool UseLayerNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for the controller.
    /// </summary>
    /// <value>
    /// Dropout rate between 0.0 and 1.0.
    /// Applied to controller layers.
    /// Default is 0.1.
    /// </value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum gradient norm.
    /// </summary>
    /// <value>
    /// Maximum norm for gradient clipping.
    /// Prevents exploding gradients.
    /// Default is 10.0.
    /// </value>
    public double MaxGradientNorm { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets whether to use memory masking.
    /// </summary>
    /// <value>
    /// If true, masks portions of memory during training.
    /// Can help prevent interference.
    /// Default is false.
    /// </value>
    public bool UseMemoryMasking { get; set; } = false;

    /// <summary>
    /// Gets or sets the memory mask ratio.
    /// </summary>
    /// <value>
    /// Fraction of memory to mask (0.0 to 1.0).
    /// Only used when UseMemoryMasking is true.
    /// Default is 0.1.
    /// </value>
    public double MemoryMaskRatio { get; set; } = 0.1;

    /// <summary>
    /// Creates a default NTM configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on the original NTM paper:
    /// - Controller: LSTM
    /// - Memory: 128 x 64
    /// - Single read/write head
    /// - Content-based addressing
    /// </remarks>
    public NTMAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 1; // NTM doesn't use inner loop
    }

    /// <summary>
    /// Creates an NTM configuration with custom values.
    /// </summary>
    /// <param name="controller">The neural network controller.</param>
    /// <param name="controllerType">Type of controller architecture.</param>
    /// <param name="memorySize">Size of external memory.</param>
    /// <param name="memoryWidth">Width of memory locations.</param>
    /// <param name="numReadHeads">Number of read heads.</param>
    /// <param name="addressingMode">Memory addressing mode.</param>
    /// <param name="initializeMemory">Whether to initialize memory.</param>
    /// <param name="memoryInitialization">Memory initialization strategy.</param>
    /// <param name="useInterpolation">Whether to use interpolation.</param>
    /// <param name="controllerHiddenUnits">Hidden units in controller.</param>
    /// <param name="controllerNumLayers">Number of controller layers.</param>
    /// <param name="innerLearningRate">Learning rate for optimization.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public NTMAlgorithmOptions(
        INeuralNetwork<T> controller,
        ControllerType controllerType = ControllerType.LSTM,
        int memorySize = 128,
        int memoryWidth = 64,
        int numReadHeads = 1,
        AddressingMode addressingMode = AddressingMode.ContentBased,
        bool initializeMemory = true,
        MemoryInitialization memoryInitialization = MemoryInitialization.Zeros,
        bool useInterpolation = true,
        int controllerHiddenUnits = 100,
        int controllerNumLayers = 1,
        double innerLearningRate = 0.001,
        int numEpisodes = 10000)
    {
        Controller = controller;
        ControllerType = controllerType;
        MemorySize = memorySize;
        MemoryWidth = memoryWidth;
        NumReadHeads = numReadHeads;
        AddressingMode = addressingMode;
        InitializeMemory = initializeMemory;
        MemoryInitialization = memoryInitialization;
        UseInterpolation = useInterpolation;
        ControllerHiddenUnits = controllerHiddenUnits;
        ControllerNumLayers = controllerNumLayers;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = 1; // NTM doesn't use inner loop
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

        // Check memory parameters
        if (MemorySize <= 0 || MemorySize > 10000)
            return false;

        if (MemoryWidth <= 0 || MemoryWidth > 1024)
            return false;

        // Check read/write heads
        if (NumReadHeads <= 0 || NumReadHeads > 10)
            return false;

        if (NumWriteHeads <= 0 || NumWriteHeads > 10)
            return false;

        // Check controller parameters
        if (ControllerHiddenUnits <= 0 || ControllerHiddenUnits > 1000)
            return false;

        if (ControllerNumLayers <= 0 || ControllerNumLayers > 10)
            return false;

        // Check regularization parameters
        if (MemoryUsageRegularization < 0.0 || MemoryUsageRegularization > 1.0)
            return false;

        if (MemorySharpnessRegularization < 0.0 || MemorySharpnessRegularization > 1.0)
            return false;

        // Check interpolation parameters
        if (UseInterpolation)
        {
            if (InterpolationStrength < 0.0 || InterpolationStrength > 1.0)
                return false;
        }

        // Check dropout rate
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            return false;

        // Check gradient norm
        if (MaxGradientNorm <= 0.0)
            return false;

        // Check memory masking
        if (UseMemoryMasking)
        {
            if (MemoryMaskRatio < 0.0 || MemoryMaskRatio > 1.0)
                return false;
        }

        // Check shift range
        if (ShiftRange < 0 || ShiftRange > MemorySize)
            return false;

        return true;
    }

    /// <summary>
    /// Gets the total number of memory parameters.
    /// </summary>
    /// <returns>Total parameters in memory matrix.</returns>
    public int GetTotalMemoryParameters()
    {
        return MemorySize * MemoryWidth;
    }

    /// <summary>
    /// Gets the total number of addressing parameters.
    /// </summary>
    /// <returns>Total parameters for read/write heads.</returns>
    public int GetTotalAddressingParameters()
    {
        // Each head needs key dimension parameters
        var keyParams = MemoryWidth * (NumReadHeads + NumWriteHeads);

        // Additional parameters for location-based addressing
        if (AddressingMode == AddressingMode.LocationBased)
        {
            keyParams += (NumReadHeads + NumWriteHeads) * 3; // shift, gamma, beta
        }
        else if (AddressingMode == AddressingMode.Mixed)
        {
            keyParams += (NumReadHeads + NumWriteHeads) * 4; // plus weighting
        }

        return keyParams;
    }
}

/// <summary>
/// Memory addressing modes for Neural Turing Machine.
/// </summary>
public enum AddressingMode
{
    /// <summary>
    /// Use content similarity for addressing.
    /// </summary>
    ContentBased,

    /// <summary>
    /// Use absolute positions for addressing.
    /// </summary>
    LocationBased,

    /// <summary>
    /// Combine content and location addressing.
    /// </summary>
    Mixed
}