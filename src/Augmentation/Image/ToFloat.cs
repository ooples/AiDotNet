namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an image tensor to floating-point representation with configurable scaling.
/// </summary>
/// <remarks>
/// <para>
/// ToFloat provides flexible conversion of image pixel values to floating-point range.
/// Unlike <see cref="ToTensor{T}"/> which always divides by 255, ToFloat allows custom
/// source and target ranges.
/// </para>
/// <para><b>For Beginners:</b> Sometimes images come in different formats: 8-bit (0-255),
/// 16-bit (0-65535), or already floating point. This transform lets you convert between
/// any of these ranges. For example, converting 16-bit medical images to [0, 1] range.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Converting 16-bit images (e.g., medical, satellite) to [0, 1]</item>
/// <item>Mapping to custom ranges like [-1, 1]</item>
/// <item>When you need more control than ToTensor provides</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ToFloat<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum value of the source range.
    /// </summary>
    public double SourceMin { get; }

    /// <summary>
    /// Gets the maximum value of the source range.
    /// </summary>
    public double SourceMax { get; }

    /// <summary>
    /// Gets the minimum value of the target range.
    /// </summary>
    public double TargetMin { get; }

    /// <summary>
    /// Gets the maximum value of the target range.
    /// </summary>
    public double TargetMax { get; }

    /// <summary>
    /// Creates a new ToFloat augmentation.
    /// </summary>
    /// <param name="sourceMin">The minimum expected input value. Default is 0.</param>
    /// <param name="sourceMax">The maximum expected input value. Default is 255.</param>
    /// <param name="targetMin">The minimum output value. Default is 0.</param>
    /// <param name="targetMax">The maximum output value. Default is 1.</param>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public ToFloat(
        double sourceMin = 0,
        double sourceMax = 255,
        double targetMin = 0,
        double targetMax = 1,
        double probability = 1.0) : base(probability)
    {
        if (Math.Abs(sourceMax - sourceMin) < 1e-10)
            throw new ArgumentException("Source range must be non-zero.");

        SourceMin = sourceMin;
        SourceMax = sourceMax;
        TargetMin = targetMin;
        TargetMax = targetMax;
    }

    /// <summary>
    /// Creates a ToFloat for 16-bit images to [0, 1].
    /// </summary>
    public static ToFloat<T> From16Bit(double probability = 1.0) =>
        new(0, 65535, 0, 1, probability);

    /// <summary>
    /// Creates a ToFloat that maps [0, 255] to [-1, 1].
    /// </summary>
    public static ToFloat<T> ToNegativeOneToOne(double probability = 1.0) =>
        new(0, 255, -1, 1, probability);

    /// <summary>
    /// Converts pixel values to the target range.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int height = data.Height;
        int width = data.Width;
        int channels = data.Channels;

        double sourceRange = SourceMax - SourceMin;
        double targetRange = TargetMax - TargetMin;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double value = NumOps.ToDouble(data.GetPixel(y, x, c));
                    // Map from source range to [0, 1], then to target range
                    double normalized = (value - SourceMin) / sourceRange;
                    double mapped = normalized * targetRange + TargetMin;
                    result.SetPixel(y, x, c, NumOps.FromDouble(mapped));
                }
            }
        }

        result.IsNormalized = (TargetMin >= -1.1 && TargetMax <= 1.1);

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["source_min"] = SourceMin;
        parameters["source_max"] = SourceMax;
        parameters["target_min"] = TargetMin;
        parameters["target_max"] = TargetMax;
        return parameters;
    }
}
