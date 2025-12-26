
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adjusts the saturation (color intensity) of an image.
/// </summary>
/// <remarks>
/// <para>
/// Saturation adjustment changes how vivid or muted colors appear in an image.
/// Higher saturation makes colors more vibrant, while lower saturation makes
/// colors appear more gray or washed out.
/// </para>
/// <para><b>For Beginners:</b> Think of saturation like the vibrancy slider on photo apps.
/// High saturation makes colors pop (like a sunset photo), while low saturation
/// makes colors look faded (like an old photograph). A saturation of 0 would make
/// the image completely grayscale.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Images from different cameras with varying color profiles</item>
/// <item>Scenes with different lighting temperatures (warm vs cool)</item>
/// <item>When you want the model to focus on shapes rather than colors</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Color is the primary classification feature (e.g., traffic lights)</item>
/// <item>Medical imaging where color accuracy matters</item>
/// <item>Tasks involving color matching or color recognition</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Saturation<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the minimum saturation adjustment factor.
    /// </summary>
    /// <remarks>
    /// A factor of 1.0 means no change. Factor of 0.0 converts to grayscale.
    /// </remarks>
    public double MinFactor { get; }

    /// <summary>
    /// Gets the maximum saturation adjustment factor.
    /// </summary>
    /// <remarks>
    /// Values greater than 1.0 increase color intensity.
    /// </remarks>
    public double MaxFactor { get; }

    /// <summary>
    /// Creates a new saturation augmentation.
    /// </summary>
    /// <param name="minFactor">
    /// The minimum saturation factor. Values below 1.0 reduce saturation.
    /// Industry standard default is 0.8 (20% desaturation at minimum).
    /// A value of 0.0 would produce a grayscale image.
    /// </param>
    /// <param name="maxFactor">
    /// The maximum saturation factor. Values above 1.0 increase saturation.
    /// Industry standard default is 1.2 (20% increase at maximum).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The factor range of 0.8 to 1.2 creates subtle,
    /// natural-looking variations. Going below 0.5 makes images noticeably washed out,
    /// while going above 2.0 can make colors unnaturally vivid.
    /// </para>
    /// </remarks>
    public Saturation(
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
    /// Applies the saturation adjustment to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Sample a random factor
        double factor = context.GetRandomDouble(MinFactor, MaxFactor);

        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Saturation only applies to color images (3+ channels)
        if (channels < 3)
        {
            return result; // Return unchanged for grayscale
        }

        // Clamp range based on whether image is normalized
        double minValue = data.IsNormalized ? 0.0 : 0.0;
        double maxValue = data.IsNormalized ? 1.0 : 255.0;

        // Standard luminance weights for RGB (ITU-R BT.601)
        const double rWeight = 0.299;
        const double gWeight = 0.587;
        const double bWeight = 0.114;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Get RGB values
                double r = Convert.ToDouble(data.GetPixel(y, x, 0));
                double g = Convert.ToDouble(data.GetPixel(y, x, 1));
                double b = Convert.ToDouble(data.GetPixel(y, x, 2));

                // Calculate luminance (grayscale equivalent)
                double gray = rWeight * r + gWeight * g + bWeight * b;

                // Interpolate between grayscale and original color based on factor
                // factor = 0: fully grayscale
                // factor = 1: original color
                // factor > 1: enhanced saturation
                double newR = gray + (r - gray) * factor;
                double newG = gray + (g - gray) * factor;
                double newB = gray + (b - gray) * factor;

                // Clamp to valid range
                newR = Math.Max(minValue, Math.Min(maxValue, newR));
                newG = Math.Max(minValue, Math.Min(maxValue, newG));
                newB = Math.Max(minValue, Math.Min(maxValue, newB));

                result.SetPixel(y, x, 0, (T)Convert.ChangeType(newR, typeof(T)));
                result.SetPixel(y, x, 1, (T)Convert.ChangeType(newG, typeof(T)));
                result.SetPixel(y, x, 2, (T)Convert.ChangeType(newB, typeof(T)));

                // Keep alpha channel unchanged if present
                for (int c = 3; c < channels; c++)
                {
                    result.SetPixel(y, x, c, data.GetPixel(y, x, c));
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
