using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for MobileNetV2 neural network architectures.
/// </summary>
/// <remarks>
/// <para>
/// MobileNetV2 is designed for efficient mobile and edge deployment, using inverted residuals
/// and linear bottlenecks to achieve high accuracy with low computational cost.
/// </para>
/// <para>
/// <b>For Beginners:</b> MobileNetV2 is optimized for mobile devices. The width multiplier
/// lets you trade accuracy for speed - smaller values give faster but less accurate models.
/// </para>
/// </remarks>
public class MobileNetV2Configuration
{
    /// <summary>
    /// Gets the width multiplier for the network.
    /// </summary>
    public MobileNetV2WidthMultiplier WidthMultiplier { get; }

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
    /// Initializes a new instance of the <see cref="MobileNetV2Configuration"/> class.
    /// </summary>
    /// <param name="widthMultiplier">The width multiplier.</param>
    /// <param name="numClasses">The number of output classes for classification.</param>
    /// <param name="inputHeight">The height of input images (default: 224).</param>
    /// <param name="inputWidth">The width of input images (default: 224).</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    public MobileNetV2Configuration(
        MobileNetV2WidthMultiplier widthMultiplier,
        int numClasses,
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
        MobileNetV2WidthMultiplier.Alpha035 => 0.35,
        MobileNetV2WidthMultiplier.Alpha050 => 0.50,
        MobileNetV2WidthMultiplier.Alpha075 => 0.75,
        MobileNetV2WidthMultiplier.Alpha100 => 1.0,
        MobileNetV2WidthMultiplier.Alpha130 => 1.3,
        MobileNetV2WidthMultiplier.Alpha140 => 1.4,
        _ => 1.0
    };

    /// <summary>
    /// Creates a MobileNetV2 configuration with standard width (recommended default).
    /// </summary>
    public static MobileNetV2Configuration CreateStandard(int numClasses)
    {
        return new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha100, numClasses);
    }
}
