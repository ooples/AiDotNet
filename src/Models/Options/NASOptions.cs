using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using System;
using System.Collections.Generic;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Neural Architecture Search (NAS).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> NAS automatically discovers optimal neural network architectures.
/// Instead of manually designing layers and connections, NAS explores the design space
/// to find architectures that maximize accuracy while meeting hardware constraints.</para>
///
/// <para><b>Quick Start Example:</b>
/// <code>
/// var options = new NASOptions&lt;double&gt;
/// {
///     Strategy = AutoMLSearchStrategy.DARTS,
///     TargetPlatform = HardwarePlatform.Mobile,
///     MaxSearchTime = TimeSpan.FromHours(2),
///     HardwareConstraints = new HardwareConstraints&lt;double&gt;
///     {
///         MaxLatency = 10.0,    // 10ms max inference time
///         MaxMemory = 50.0      // 50MB max model size
///     }
/// };
/// </code>
/// </para>
///
/// <para><b>Available Strategies:</b>
/// <list type="bullet">
/// <item><description><b>DARTS:</b> Fast gradient-based search (~1-2 GPU days)</description></item>
/// <item><description><b>GDAS:</b> Improved DARTS with better discretization</description></item>
/// <item><description><b>OnceForAll:</b> Train once, deploy anywhere with elastic networks</description></item>
/// <item><description><b>NeuralArchitectureSearch:</b> Auto-selects best algorithm</description></item>
/// </list>
/// </para>
/// </remarks>
public class NASOptions<T>
{
    /// <summary>
    /// Gets or sets the NAS strategy to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different strategies have different trade-offs:
    /// <list type="bullet">
    /// <item><description><b>DARTS:</b> Fast but may produce suboptimal architectures</description></item>
    /// <item><description><b>GDAS:</b> More stable than DARTS, slightly slower</description></item>
    /// <item><description><b>OnceForAll:</b> Best for multi-device deployment</description></item>
    /// </list></para>
    /// </remarks>
    public AutoMLSearchStrategy Strategy { get; set; } = AutoMLSearchStrategy.DARTS;

    /// <summary>
    /// Gets or sets the target hardware platform for optimization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choose where your model will run:
    /// <list type="bullet">
    /// <item><description><b>CPU:</b> Standard desktop/server deployment</description></item>
    /// <item><description><b>GPU:</b> High-performance with CUDA/OpenCL</description></item>
    /// <item><description><b>Mobile:</b> Smartphones and tablets</description></item>
    /// <item><description><b>EdgeTPU:</b> Google Edge TPU accelerators</description></item>
    /// </list></para>
    /// </remarks>
    public HardwarePlatform TargetPlatform { get; set; } = HardwarePlatform.CPU;

    /// <summary>
    /// Gets or sets the hardware constraints for the architecture search.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set limits that your target device can handle:
    /// <code>
    /// HardwareConstraints = new HardwareConstraints&lt;double&gt;
    /// {
    ///     MaxLatency = 10.0,    // Max inference time in milliseconds
    ///     MaxMemory = 50.0,     // Max model size in MB
    ///     MaxEnergy = 100.0     // Max energy per inference in mJ
    /// };
    /// </code></para>
    /// </remarks>
    public HardwareConstraints<T>? HardwareConstraints { get; set; }

    /// <summary>
    /// Gets or sets the search space configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Customize the operations and structure that NAS can explore.
    /// If null, a default search space with standard operations is used.</para>
    /// </remarks>
    public SearchSpaceBase<T>? SearchSpace { get; set; }

    /// <summary>
    /// Gets or sets the maximum time for the architecture search.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Longer search times generally find better architectures.
    /// Typical values:
    /// <list type="bullet">
    /// <item><description><b>Quick test:</b> 30 minutes - 1 hour</description></item>
    /// <item><description><b>Standard:</b> 2-8 hours</description></item>
    /// <item><description><b>Production:</b> 24-48 hours</description></item>
    /// </list></para>
    /// </remarks>
    public TimeSpan MaxSearchTime { get; set; } = TimeSpan.FromHours(8);

    /// <summary>
    /// Gets or sets the maximum number of search epochs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of training iterations for the architecture search.
    /// Higher values explore more thoroughly but take longer.</para>
    /// </remarks>
    public int MaxEpochs { get; set; } = 50;

    /// <summary>
    /// Gets or sets the learning rate for architecture parameters.
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Controls how fast the architecture search converges.
    /// Lower values are more stable but slower.</para>
    /// </remarks>
    public double ArchitectureLearningRate { get; set; } = 0.003;

    /// <summary>
    /// Gets or sets the learning rate for network weights.
    /// </summary>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Controls how fast the network weights update.
    /// Typically higher than architecture learning rate.</para>
    /// </remarks>
    public double WeightLearningRate { get; set; } = 0.025;

    // ============ Once-for-All (OFA) Specific Settings ============

    /// <summary>
    /// Gets or sets the elastic depth values for OFA networks.
    /// </summary>
    /// <remarks>
    /// <para><b>For OFA:</b> List of possible layer counts that the network can use.
    /// Example: [2, 3, 4] means network can have 2, 3, or 4 layers per block.</para>
    /// </remarks>
    public List<int>? ElasticDepths { get; set; }

    /// <summary>
    /// Gets or sets the elastic width multipliers for OFA networks.
    /// </summary>
    /// <remarks>
    /// <para><b>For OFA:</b> List of channel width multipliers.
    /// Example: [0.75, 1.0, 1.25] means 75%, 100%, or 125% of base channels.</para>
    /// </remarks>
    public List<double>? ElasticWidths { get; set; }

    /// <summary>
    /// Gets or sets the elastic kernel sizes for OFA networks.
    /// </summary>
    /// <remarks>
    /// <para><b>For OFA:</b> List of possible kernel sizes.
    /// Example: [3, 5, 7] means 3x3, 5x5, or 7x7 kernels.</para>
    /// </remarks>
    public List<int>? ElasticKernelSizes { get; set; }

    /// <summary>
    /// Gets or sets the elastic expansion ratios for OFA inverted residuals.
    /// </summary>
    /// <remarks>
    /// <para><b>For OFA:</b> List of expansion ratios for inverted residual blocks.
    /// Example: [3, 4, 6] for MobileNetV2-style blocks.</para>
    /// </remarks>
    public List<int>? ElasticExpansionRatios { get; set; }

    // ============ Evolutionary Search Settings ============

    /// <summary>
    /// Gets or sets the population size for evolutionary search.
    /// </summary>
    /// <remarks>
    /// <para><b>For Evolutionary/OFA Specialization:</b> Number of candidate architectures
    /// to maintain in each generation. Larger populations explore more diversity.</para>
    /// </remarks>
    public int PopulationSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of generations for evolutionary search.
    /// </summary>
    /// <remarks>
    /// <para><b>For Evolutionary/OFA Specialization:</b> Number of evolution cycles.
    /// More generations allow better convergence.</para>
    /// </remarks>
    public int Generations { get; set; } = 50;

    /// <summary>
    /// Gets or sets the mutation probability for evolutionary search.
    /// </summary>
    /// <remarks>
    /// <para><b>For Evolutionary:</b> Probability of mutating each gene in offspring.
    /// Typical values: 0.05-0.2</para>
    /// </remarks>
    public double MutationProbability { get; set; } = 0.1;

    // ============ Quantization-Aware Settings ============

    /// <summary>
    /// Gets or sets whether to use quantization-aware training during NAS.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, architectures are searched with quantization
    /// in mind, resulting in models that maintain accuracy after quantization.</para>
    /// </remarks>
    public bool QuantizationAware { get; set; } = false;

    /// <summary>
    /// Gets or sets the quantization mode for quantization-aware NAS.
    /// </summary>
    /// <remarks>
    /// <para><b>For Quantization-Aware NAS:</b> The type of quantization to simulate:
    /// <list type="bullet">
    /// <item><description><b>Int8:</b> Most common for mobile/edge deployment</description></item>
    /// <item><description><b>Float16:</b> Good balance of speed and accuracy</description></item>
    /// <item><description><b>Dynamic:</b> Quantize weights, compute in float</description></item>
    /// <item><description><b>Mixed:</b> Different precision for different layers</description></item>
    /// </list></para>
    /// </remarks>
    public QuantizationMode QuantizationMode { get; set; } = QuantizationMode.Int8;

    // ============ Input/Output Configuration ============

    /// <summary>
    /// Gets or sets the number of input channels.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of channels in input data:
    /// <list type="bullet">
    /// <item><description><b>1:</b> Grayscale images</description></item>
    /// <item><description><b>3:</b> RGB images</description></item>
    /// <item><description><b>4:</b> RGBA images</description></item>
    /// </list></para>
    /// </remarks>
    public int InputChannels { get; set; } = 3;

    /// <summary>
    /// Gets or sets the spatial size of the input (assuming square inputs).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Width/height of input images.
    /// Common values: 32 (CIFAR), 224 (ImageNet), 299 (Inception).</para>
    /// </remarks>
    public int SpatialSize { get; set; } = 224;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Classification:</b> Number of classes to predict.
    /// Example: 10 for CIFAR-10, 1000 for ImageNet.</para>
    /// </remarks>
    public int NumClasses { get; set; } = 10;

    // ============ Callbacks and Monitoring ============

    /// <summary>
    /// Gets or sets the callback invoked after each search epoch.
    /// </summary>
    /// <remarks>
    /// <para><b>For Monitoring:</b> Use to log progress or save checkpoints.
    /// Parameters are (epoch, current best score).</para>
    /// </remarks>
    public Action<int, T>? OnEpochComplete { get; set; }

    /// <summary>
    /// Gets or sets whether to save architecture checkpoints during search.
    /// </summary>
    public bool SaveCheckpoints { get; set; } = false;

    /// <summary>
    /// Gets or sets the checkpoint directory path.
    /// </summary>
    public string? CheckpointDirectory { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <para><b>For Reproducibility:</b> Set a seed to get repeatable results.</para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets whether to enable verbose logging during search.
    /// </summary>
    public bool Verbose { get; set; } = true;

    /// <summary>
    /// Validates the options and throws if any are invalid.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when options are invalid.</exception>
    public void Validate()
    {
        if (MaxEpochs <= 0)
            throw new ArgumentException("MaxEpochs must be positive", nameof(MaxEpochs));

        if (MaxSearchTime <= TimeSpan.Zero)
            throw new ArgumentException("MaxSearchTime must be positive", nameof(MaxSearchTime));

        if (ArchitectureLearningRate <= 0)
            throw new ArgumentException("ArchitectureLearningRate must be positive", nameof(ArchitectureLearningRate));

        if (WeightLearningRate <= 0)
            throw new ArgumentException("WeightLearningRate must be positive", nameof(WeightLearningRate));

        if (PopulationSize < 2)
            throw new ArgumentException("PopulationSize must be at least 2", nameof(PopulationSize));

        if (Generations <= 0)
            throw new ArgumentException("Generations must be positive", nameof(Generations));

        if (MutationProbability < 0 || MutationProbability > 1)
            throw new ArgumentException("MutationProbability must be between 0 and 1", nameof(MutationProbability));

        if (InputChannels <= 0)
            throw new ArgumentException("InputChannels must be positive", nameof(InputChannels));

        if (SpatialSize <= 0)
            throw new ArgumentException("SpatialSize must be positive", nameof(SpatialSize));

        if (NumClasses <= 0)
            throw new ArgumentException("NumClasses must be positive", nameof(NumClasses));

        // Validate strategy is a valid NAS strategy
        var validNasStrategies = new[]
        {
            AutoMLSearchStrategy.NeuralArchitectureSearch,
            AutoMLSearchStrategy.DARTS,
            AutoMLSearchStrategy.GDAS,
            AutoMLSearchStrategy.OnceForAll
        };

        if (!Array.Exists(validNasStrategies, s => s == Strategy))
        {
            throw new ArgumentException(
                $"Strategy must be a NAS strategy (DARTS, GDAS, OnceForAll, or NeuralArchitectureSearch). Got: {Strategy}",
                nameof(Strategy));
        }

        // Validate OFA-specific settings if using OnceForAll
        if (Strategy == AutoMLSearchStrategy.OnceForAll)
        {
            if (ElasticDepths != null && ElasticDepths.Count == 0)
                throw new ArgumentException("ElasticDepths cannot be empty if provided", nameof(ElasticDepths));

            if (ElasticWidths != null && ElasticWidths.Count == 0)
                throw new ArgumentException("ElasticWidths cannot be empty if provided", nameof(ElasticWidths));

            if (ElasticKernelSizes != null && ElasticKernelSizes.Count == 0)
                throw new ArgumentException("ElasticKernelSizes cannot be empty if provided", nameof(ElasticKernelSizes));

            if (ElasticExpansionRatios != null && ElasticExpansionRatios.Count == 0)
                throw new ArgumentException("ElasticExpansionRatios cannot be empty if provided", nameof(ElasticExpansionRatios));
        }
    }

    /// <summary>
    /// Creates a copy of these options with the specified modifications.
    /// </summary>
    /// <param name="configure">Action to configure the copy.</param>
    /// <returns>A new NASOptions instance with modifications applied.</returns>
    public NASOptions<T> With(Action<NASOptions<T>> configure)
    {
        var copy = new NASOptions<T>
        {
            Strategy = Strategy,
            TargetPlatform = TargetPlatform,
            HardwareConstraints = HardwareConstraints,
            SearchSpace = SearchSpace,
            MaxSearchTime = MaxSearchTime,
            MaxEpochs = MaxEpochs,
            ArchitectureLearningRate = ArchitectureLearningRate,
            WeightLearningRate = WeightLearningRate,
            ElasticDepths = ElasticDepths != null ? new List<int>(ElasticDepths) : null,
            ElasticWidths = ElasticWidths != null ? new List<double>(ElasticWidths) : null,
            ElasticKernelSizes = ElasticKernelSizes != null ? new List<int>(ElasticKernelSizes) : null,
            ElasticExpansionRatios = ElasticExpansionRatios != null ? new List<int>(ElasticExpansionRatios) : null,
            PopulationSize = PopulationSize,
            Generations = Generations,
            MutationProbability = MutationProbability,
            QuantizationAware = QuantizationAware,
            QuantizationMode = QuantizationMode,
            InputChannels = InputChannels,
            SpatialSize = SpatialSize,
            NumClasses = NumClasses,
            OnEpochComplete = OnEpochComplete,
            SaveCheckpoints = SaveCheckpoints,
            CheckpointDirectory = CheckpointDirectory,
            RandomSeed = RandomSeed,
            Verbose = Verbose
        };

        configure(copy);
        return copy;
    }
}
