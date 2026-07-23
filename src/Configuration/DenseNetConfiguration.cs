using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for DenseNet neural network architectures.
/// </summary>
/// <remarks>
/// <para>
/// DenseNet (Densely Connected Convolutional Networks) connects each layer to every other layer
/// in a feed-forward fashion, enabling strong gradient flow and feature reuse.
/// </para>
/// <para>
/// <b>For Beginners:</b> DenseNet is designed to maximize information flow by connecting each
/// layer directly to all subsequent layers. This configuration lets you choose which variant
/// to use and customize parameters like growth rate and compression factor.
/// </para>
/// </remarks>
public class DenseNetConfiguration
{
    /// <summary>
    /// Gets the DenseNet variant to use.
    /// </summary>
    public DenseNetVariant Variant { get; }

    /// <summary>
    /// Gets the number of output classes for classification.
    /// </summary>
    public int NumClasses { get; }

    /// <summary>
    /// Gets the height of input images in pixels.
    /// </summary>
    public int InputHeight { get; }

    /// <summary>
    /// Gets the width of input images in pixels.
    /// </summary>
    public int InputWidth { get; }

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InputChannels { get; }

    /// <summary>
    /// Gets the growth rate (k in the paper).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The growth rate determines how many new feature maps each layer adds.
    /// Typical values are 12, 24, or 32. Higher values increase capacity but also computational cost.
    /// </para>
    /// </remarks>
    public int GrowthRate { get; }

    /// <summary>
    /// Gets the compression factor for transition layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Compression factor (theta) controls channel reduction at transition layers.
    /// A value of 0.5 means halving the channels at each transition, which helps control model size.
    /// </para>
    /// </remarks>
    public double CompressionFactor { get; }

    /// <summary>
    /// Gets the custom block layers configuration (only used when Variant is Custom).
    /// </summary>
    public int[]? CustomBlockLayers { get; }

    /// <summary>
    /// Gets the computed input shape as [channels, height, width].
    /// </summary>
    public int[] InputShape => [InputChannels, InputHeight, InputWidth];

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseNetConfiguration"/> class.
    /// </summary>
    /// <param name="variant">The DenseNet variant to use.</param>
    /// <param name="numClasses">The number of output classes for classification.</param>
    /// <param name="inputHeight">The height of input images (default: 224).</param>
    /// <param name="inputWidth">The width of input images (default: 224).</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <param name="growthRate">The growth rate (default: 32).</param>
    /// <param name="compressionFactor">The compression factor for transition layers (default: 0.5).</param>
    /// <param name="customBlockLayers">Custom block layers (required when variant is Custom).</param>
    public DenseNetConfiguration(
        DenseNetVariant variant,
        int numClasses,
        int inputHeight = 224,
        int inputWidth = 224,
        int inputChannels = 3,
        int growthRate = 32,
        double compressionFactor = 0.5,
        int[]? customBlockLayers = null)
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be greater than 0.");
        if (inputHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputHeight), "Input height must be greater than 0.");
        if (inputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be greater than 0.");
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be greater than 0.");
        if (growthRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(growthRate), "Growth rate must be greater than 0.");
        if (compressionFactor <= 0 || compressionFactor > 1)
            throw new ArgumentOutOfRangeException(nameof(compressionFactor), "Compression factor must be between 0 (exclusive) and 1 (inclusive).");
        if (variant == DenseNetVariant.Custom && (customBlockLayers == null || customBlockLayers.Length == 0))
            throw new ArgumentException("Custom block layers must be provided when using Custom variant.", nameof(customBlockLayers));

        Variant = variant;
        NumClasses = numClasses;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        InputChannels = inputChannels;
        GrowthRate = growthRate;
        CompressionFactor = compressionFactor;
        CustomBlockLayers = customBlockLayers;
    }

    /// <summary>
    /// Gets the number of layers per dense block for this variant.
    /// </summary>
    public int[] GetBlockLayers()
    {
        return Variant switch
        {
            DenseNetVariant.DenseNet121 => [6, 12, 24, 16],
            DenseNetVariant.DenseNet169 => [6, 12, 32, 32],
            DenseNetVariant.DenseNet201 => [6, 12, 48, 32],
            DenseNetVariant.DenseNet264 => [6, 12, 64, 48],
            DenseNetVariant.Custom => CustomBlockLayers ?? [2, 2, 2, 2],
            _ => [6, 12, 24, 16]
        };
    }

    /// <summary>
    /// Gets the expected total layer count for this configuration without constructing the network.
    /// </summary>
    /// <remarks>
    /// This is useful for tests that need to compare layer counts without the overhead
    /// of actually constructing the networks.
    /// Formula: 1 (stem conv) + 1 (stem BN) + 1 (stem pool) +
    ///          sum(block_layers * 2) for BN+Conv in each dense layer +
    ///          (num_blocks - 1) * 2 for transition layers (Conv + Pool) +
    ///          1 (final BN) + 1 (classifier)
    /// </remarks>
    /// <returns>The expected number of layers in the network.</returns>
    public int GetExpectedLayerCount()
    {
        var blockLayers = GetBlockLayers();

        // Stem: Conv + BN + Pool = 3 layers
        int stemLayers = 3;

        // Each dense block layer has: BN + Conv (we count each as separate layer)
        // In DenseNet, each "layer" in a block is actually BN-ReLU-Conv (bottleneck may add more)
        // For simplicity, count 2 layers per block layer (this is an approximation)
        int denseBlockLayers = 0;
        foreach (var layers in blockLayers)
        {
            denseBlockLayers += layers * 2; // BN + Conv per layer
        }

        // Transition layers between blocks: (numBlocks - 1) * 2 (BN + Conv + Pool counted as 2)
        int transitionLayers = (blockLayers.Length - 1) * 2;

        // Final: BN + Classifier = 2 layers
        int finalLayers = 2;

        return stemLayers + denseBlockLayers + transitionLayers + finalLayers;
    }

    /// <summary>
    /// Creates a DenseNet-121 configuration (recommended default).
    /// </summary>
    public static DenseNetConfiguration CreateDenseNet121(int numClasses)
    {
        return new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses);
    }

    /// <summary>
    /// Creates a minimal DenseNet configuration optimized for fast test execution.
    /// </summary>
    /// <remarks>
    /// Uses [2, 2, 2, 2] block configuration with small growth rate (8) and 32x32 input,
    /// resulting in approximately 8 dense layers instead of 58+ in DenseNet-121.
    /// Construction time is typically under 50ms.
    /// </remarks>
    /// <param name="numClasses">The number of output classes.</param>
    /// <returns>A minimal DenseNet configuration for testing.</returns>
    public static DenseNetConfiguration CreateForTesting(int numClasses)
    {
        return new DenseNetConfiguration(
            variant: DenseNetVariant.Custom,
            numClasses: numClasses,
            inputHeight: 32,
            inputWidth: 32,
            inputChannels: 3,
            growthRate: 8,
            compressionFactor: 0.5,
            customBlockLayers: [2, 2, 2, 2]);
    }
}
