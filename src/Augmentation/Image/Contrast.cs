
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adjusts the contrast of an image by scaling pixel values around the mean.
/// </summary>
/// <remarks>
/// <para>
/// Contrast adjustment changes the difference between light and dark areas of an image.
/// Higher contrast makes light areas lighter and dark areas darker, while lower contrast
/// makes the image appear more "washed out" or flat.
/// </para>
/// <para><b>For Beginners:</b> Think of contrast like the difference between a sunny day
/// (high contrast with bright lights and dark shadows) and a foggy day (low contrast where
/// everything looks similar in brightness). This teaches your model to recognize objects
/// in both crisp and hazy conditions.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Images from cameras with different quality or settings</item>
/// <item>Scenes with varying lighting conditions</item>
/// <item>Data from different sources with different post-processing</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Tasks where pixel intensity relationships are critical</item>
/// <item>Medical imaging where contrast carries diagnostic information</item>
/// <item>Already preprocessed images with standardized contrast</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Contrast<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the minimum contrast adjustment factor.
    /// </summary>
    /// <remarks>
    /// A factor of 1.0 means no change. Values less than 1.0 reduce contrast.
    /// </remarks>
    public double MinFactor { get; }

    /// <summary>
    /// Gets the maximum contrast adjustment factor.
    /// </summary>
    /// <remarks>
    /// Values greater than 1.0 increase contrast.
    /// </remarks>
    public double MaxFactor { get; }

    /// <summary>
    /// Creates a new contrast augmentation.
    /// </summary>
    /// <param name="minFactor">
    /// The minimum contrast factor. Values below 1.0 reduce contrast.
    /// Industry standard default is 0.8 (20% reduction at minimum).
    /// </param>
    /// <param name="maxFactor">
    /// The maximum contrast factor. Values above 1.0 increase contrast.
    /// Industry standard default is 1.2 (20% increase at maximum).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The factor range of 0.8 to 1.2 creates natural-looking
    /// variations. Factor of 0.0 would make all pixels the same gray, while very high
    /// factors (like 3.0) would create harsh, high-contrast images.
    /// </para>
    /// </remarks>
    public Contrast(
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
    /// Applies the contrast adjustment to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Sample a random factor
        double factor = context.GetRandomDouble(MinFactor, MaxFactor);

        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Calculate the mean of the image (for each channel or overall)
        double[] channelMeans = new double[channels];
        int totalPixels = height * width;

        for (int c = 0; c < channels; c++)
        {
            double sum = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    sum += NumOps.ToDouble(data.GetPixel(y, x, c));
                }
            }
            channelMeans[c] = sum / totalPixels;
        }

        // Clamp range based on whether image is normalized
        double minValue = data.IsNormalized ? 0.0 : 0.0;
        double maxValue = data.IsNormalized ? 1.0 : 255.0;

        // Apply contrast adjustment: new_pixel = mean + (pixel - mean) * factor
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double value = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double adjusted = channelMeans[c] + (value - channelMeans[c]) * factor;

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
