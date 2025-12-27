using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for MobileNetV3 neural network architectures.
/// </summary>
/// <remarks>
/// <para>
/// MobileNetV3 builds on MobileNetV2 with additional optimizations including
/// squeeze-and-excitation blocks and hard-swish activation for improved accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> MobileNetV3 comes in Large (more accurate) and Small (faster)
/// variants. Choose based on whether you prioritize accuracy or speed.
/// </para>
/// </remarks>
public class MobileNetV3Configuration
{
    /// <summary>
    /// Gets the MobileNetV3 variant to use.
    /// </summary>
    public MobileNetV3Variant Variant { get; }

    /// <summary>
    /// Gets the width multiplier for the network.
    /// </summary>
    public MobileNetV3WidthMultiplier WidthMultiplier { get; }

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
    /// Gets the computed input shape as [channels, height, width].
    /// </summary>
    public int[] InputShape => [InputChannels, InputHeight, InputWidth];

    /// <summary>
    /// Initializes a new instance of the <see cref="MobileNetV3Configuration"/> class.
    /// </summary>
    /// <param name="variant">The MobileNetV3 variant (Large or Small).</param>
    /// <param name="numClasses">The number of output classes for classification.</param>
    /// <param name="widthMultiplier">The width multiplier (default: Alpha100).</param>
    /// <param name="inputHeight">The height of input images (default: 224).</param>
    /// <param name="inputWidth">The width of input images (default: 224).</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    public MobileNetV3Configuration(
        MobileNetV3Variant variant,
        int numClasses,
        MobileNetV3WidthMultiplier widthMultiplier = MobileNetV3WidthMultiplier.Alpha100,
        int inputHeight = 224,
        int inputWidth = 224,
        int inputChannels = 3)
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be greater than 0.");
        if (inputHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputHeight), "Input height must be greater than 0.");
        if (inputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be greater than 0.");
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be greater than 0.");

        Variant = variant;
        WidthMultiplier = widthMultiplier;
        NumClasses = numClasses;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        InputChannels = inputChannels;
    }

    /// <summary>
    /// Gets the alpha value (width multiplier as a double).
    /// </summary>
    public double Alpha => WidthMultiplier switch
    {
        MobileNetV3WidthMultiplier.Alpha075 => 0.75,
        MobileNetV3WidthMultiplier.Alpha100 => 1.0,
        _ => 1.0
    };

    /// <summary>
    /// Creates a MobileNetV3-Large configuration (recommended default).
    /// </summary>
    public static MobileNetV3Configuration CreateLarge(int numClasses)
    {
        return new MobileNetV3Configuration(MobileNetV3Variant.Large, numClasses);
    }

    /// <summary>
    /// Creates a MobileNetV3-Small configuration for low-latency applications.
    /// </summary>
    public static MobileNetV3Configuration CreateSmall(int numClasses)
    {
        return new MobileNetV3Configuration(MobileNetV3Variant.Small, numClasses);
    }
}
