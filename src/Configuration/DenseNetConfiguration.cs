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
    public DenseNetConfiguration(
        DenseNetVariant variant,
        int numClasses,
        int inputHeight = 224,
        int inputWidth = 224,
        int inputChannels = 3,
        int growthRate = 32,
        double compressionFactor = 0.5)
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

        Variant = variant;
        NumClasses = numClasses;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        InputChannels = inputChannels;
        GrowthRate = growthRate;
        CompressionFactor = compressionFactor;
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
            _ => [6, 12, 24, 16]
        };
    }

    /// <summary>
    /// Creates a DenseNet-121 configuration (recommended default).
    /// </summary>
    public static DenseNetConfiguration CreateDenseNet121(int numClasses)
    {
        return new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses);
    }
}
