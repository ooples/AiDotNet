namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adjusts the brightness of an image by adding a random offset to all pixel values.
/// </summary>
/// <remarks>
/// <para>
/// Brightness adjustment simulates different lighting conditions by uniformly increasing
/// or decreasing all pixel values. This helps models become robust to variations in
/// ambient lighting and exposure settings.
/// </para>
/// <para><b>For Beginners:</b> Think of this like adjusting the brightness slider on your phone.
/// Making an image brighter adds light to all pixels, making it darker subtracts light.
/// This teaches your model to recognize objects whether they're in bright sunlight or shade.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Outdoor photography where lighting varies by time of day</item>
/// <item>Indoor scenes with different lighting conditions</item>
/// <item>Any task where exposure/lighting might vary between training and deployment</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Tasks where absolute brightness is meaningful (e.g., astronomy)</item>
/// <item>Images already normalized to a specific brightness range</item>
/// <item>Medical imaging where pixel intensity has diagnostic meaning</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Brightness<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the minimum brightness adjustment factor.
    /// </summary>
    /// <remarks>
    /// A factor of 1.0 means no change. Values less than 1.0 darken the image.
    /// </remarks>
    public double MinFactor { get; }

    /// <summary>
    /// Gets the maximum brightness adjustment factor.
    /// </summary>
    /// <remarks>
    /// Values greater than 1.0 brighten the image.
    /// </remarks>
    public double MaxFactor { get; }

    /// <summary>
    /// Creates a new brightness augmentation.
    /// </summary>
    /// <param name="minFactor">
    /// The minimum brightness factor. Values below 1.0 darken the image.
    /// Industry standard default is 0.8 (20% darker at minimum).
    /// </param>
    /// <param name="maxFactor">
    /// The maximum brightness factor. Values above 1.0 brighten the image.
    /// Industry standard default is 1.2 (20% brighter at maximum).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The factor range of 0.8 to 1.2 creates subtle brightness
    /// variations that look natural. A range like 0.5 to 1.5 would create more dramatic changes.
    /// Factor of 0.0 would make the image completely black, factor of 2.0 would make it very bright.
    /// </para>
    /// </remarks>
    public Brightness(
        double minFactor = 0.8,
        double maxFactor = 1.2,
        double probability = 0.5) : base(probability)
    {
        if (minFactor < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minFactor), "minFactor must be non-negative");
        }
        if (minFactor > maxFactor)
        {
            throw new ArgumentException("minFactor must be less than or equal to maxFactor", nameof(minFactor));
        }

        MinFactor = minFactor;
        MaxFactor = maxFactor;
    }

    /// <summary>
    /// Applies the brightness adjustment to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Sample a random factor
        double factor = context.GetRandomDouble(MinFactor, MaxFactor);

        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Apply brightness adjustment: new_pixel = pixel * factor
        // Clamp to valid range based on whether image is normalized
        double minValue = data.IsNormalized ? 0.0 : 0.0;
        double maxValue = data.IsNormalized ? 1.0 : 255.0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double value = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double adjusted = value * factor;

                    // Clamp to valid range
                    adjusted = Math.Max(minValue, Math.Min(maxValue, adjusted));

                    result.SetPixel(y, x, c, NumOps.FromDouble(adjusted));
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["min_factor"] = MinFactor;
        parameters["max_factor"] = MaxFactor;
        return parameters;
    }
}
