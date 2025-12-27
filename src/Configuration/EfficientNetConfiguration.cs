using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for EfficientNet neural network architectures.
/// </summary>
/// <remarks>
/// <para>
/// EfficientNet uses compound scaling to balance network depth, width, and resolution.
/// Each variant (B0-B7) represents a different scale factor.
/// </para>
/// <para>
/// <b>For Beginners:</b> EfficientNet is designed to achieve better accuracy with fewer
/// parameters by systematically scaling all network dimensions. Choose a variant based
/// on your accuracy requirements and computational budget.
/// </para>
/// </remarks>
public class EfficientNetConfiguration
{
    /// <summary>
    /// Gets the EfficientNet variant to use.
    /// </summary>
    public EfficientNetVariant Variant { get; }

    /// <summary>
    /// Gets the number of output classes for classification.
    /// </summary>
    public int NumClasses { get; }

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InputChannels { get; }

    /// <summary>
    /// Gets the computed input shape as [channels, height, width].
    /// </summary>
    public int[] InputShape => [InputChannels, GetInputHeight(), GetInputWidth()];

    /// <summary>
    /// Initializes a new instance of the <see cref="EfficientNetConfiguration"/> class.
    /// </summary>
    /// <param name="variant">The EfficientNet variant to use.</param>
    /// <param name="numClasses">The number of output classes for classification.</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    public EfficientNetConfiguration(
        EfficientNetVariant variant,
        int numClasses,
        int inputChannels = 3)
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be greater than 0.");
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be greater than 0.");

        Variant = variant;
        NumClasses = numClasses;
        InputChannels = inputChannels;
    }

    /// <summary>
    /// Gets the recommended input height for this variant.
    /// </summary>
    public int GetInputHeight()
    {
        return Variant switch
        {
            EfficientNetVariant.B0 => 224,
            EfficientNetVariant.B1 => 240,
            EfficientNetVariant.B2 => 260,
            EfficientNetVariant.B3 => 300,
            EfficientNetVariant.B4 => 380,
            EfficientNetVariant.B5 => 456,
            EfficientNetVariant.B6 => 528,
            EfficientNetVariant.B7 => 600,
            _ => 224
        };
    }

    /// <summary>
    /// Gets the recommended input width for this variant.
    /// </summary>
    public int GetInputWidth() => GetInputHeight();

    /// <summary>
    /// Gets the width multiplier for this variant.
    /// </summary>
    public double GetWidthMultiplier()
    {
        return Variant switch
        {
            EfficientNetVariant.B0 => 1.0,
            EfficientNetVariant.B1 => 1.0,
            EfficientNetVariant.B2 => 1.1,
            EfficientNetVariant.B3 => 1.2,
            EfficientNetVariant.B4 => 1.4,
            EfficientNetVariant.B5 => 1.6,
            EfficientNetVariant.B6 => 1.8,
            EfficientNetVariant.B7 => 2.0,
            _ => 1.0
        };
    }

    /// <summary>
    /// Gets the depth multiplier for this variant.
    /// </summary>
    public double GetDepthMultiplier()
    {
        return Variant switch
        {
            EfficientNetVariant.B0 => 1.0,
            EfficientNetVariant.B1 => 1.1,
            EfficientNetVariant.B2 => 1.2,
            EfficientNetVariant.B3 => 1.4,
            EfficientNetVariant.B4 => 1.8,
            EfficientNetVariant.B5 => 2.2,
            EfficientNetVariant.B6 => 2.6,
            EfficientNetVariant.B7 => 3.1,
            _ => 1.0
        };
    }

    /// <summary>
    /// Gets the dropout rate for this variant.
    /// </summary>
    public double GetDropoutRate()
    {
        return Variant switch
        {
            EfficientNetVariant.B0 => 0.2,
            EfficientNetVariant.B1 => 0.2,
            EfficientNetVariant.B2 => 0.3,
            EfficientNetVariant.B3 => 0.3,
            EfficientNetVariant.B4 => 0.4,
            EfficientNetVariant.B5 => 0.4,
            EfficientNetVariant.B6 => 0.5,
            EfficientNetVariant.B7 => 0.5,
            _ => 0.2
        };
    }

    /// <summary>
    /// Creates an EfficientNet-B0 configuration (recommended default).
    /// </summary>
    public static EfficientNetConfiguration CreateB0(int numClasses)
    {
        return new EfficientNetConfiguration(EfficientNetVariant.B0, numClasses);
    }
}
