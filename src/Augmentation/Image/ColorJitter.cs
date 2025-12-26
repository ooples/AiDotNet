
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies random combinations of brightness, contrast, saturation, and hue adjustments.
/// </summary>
/// <remarks>
/// <para>
/// ColorJitter is a powerful composite augmentation that randomly adjusts multiple color
/// properties in a single operation. This simulates the wide variety of color variations
/// that occur in real-world photography due to different cameras, lighting, and environments.
/// </para>
/// <para><b>For Beginners:</b> Think of this as applying multiple photo filters randomly.
/// Just like how the same scene looks different when photographed with different phones
/// or in different lighting, ColorJitter creates these natural variations automatically.
/// This is one of the most commonly used augmentations for image classification.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>General image classification tasks</item>
/// <item>When training data comes from a single camera/source but deployment varies</item>
/// <item>To make models robust to different lighting and camera settings</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Color is the primary classification feature</item>
/// <item>Medical/scientific imaging with calibrated color</item>
/// <item>Tasks where specific color accuracy is required</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ColorJitter<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the brightness adjustment range (0.0 = no change possible).
    /// </summary>
    public double BrightnessRange { get; }

    /// <summary>
    /// Gets the contrast adjustment range (0.0 = no change possible).
    /// </summary>
    public double ContrastRange { get; }

    /// <summary>
    /// Gets the saturation adjustment range (0.0 = no change possible).
    /// </summary>
    public double SaturationRange { get; }

    /// <summary>
    /// Gets the hue adjustment range in degrees (0.0 = no change possible).
    /// </summary>
    public double HueRange { get; }

    /// <summary>
    /// Creates a new color jitter augmentation.
    /// </summary>
    /// <param name="brightnessRange">
    /// The range for brightness adjustment. The actual factor will be sampled from
    /// [1 - brightnessRange, 1 + brightnessRange]. Set to 0.0 to disable.
    /// Industry standard default is 0.2 (factors from 0.8 to 1.2).
    /// </param>
    /// <param name="contrastRange">
    /// The range for contrast adjustment. The actual factor will be sampled from
    /// [1 - contrastRange, 1 + contrastRange]. Set to 0.0 to disable.
    /// Industry standard default is 0.2 (factors from 0.8 to 1.2).
    /// </param>
    /// <param name="saturationRange">
    /// The range for saturation adjustment. The actual factor will be sampled from
    /// [1 - saturationRange, 1 + saturationRange]. Set to 0.0 to disable.
    /// Industry standard default is 0.2 (factors from 0.8 to 1.2).
    /// </param>
    /// <param name="hueRange">
    /// The range for hue adjustment in degrees. The actual shift will be sampled from
    /// [-hueRange, +hueRange]. Set to 0.0 to disable.
    /// Industry standard default is 0.1 (about 36 degrees maximum shift).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.8 (applied to 80% of images).
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each range parameter controls how much that property
    /// can vary. A brightnessRange of 0.2 means images can be 20% brighter or darker.
    /// The hueRange is in degrees (0-1 representing 0-360 degrees) - use small values
    /// like 0.1 to avoid unnatural color shifts.
    /// </para>
    /// </remarks>
    public ColorJitter(
        double brightnessRange = 0.2,
        double contrastRange = 0.2,
        double saturationRange = 0.2,
        double hueRange = 0.1,
        double probability = 0.8) : base(probability)
    {
        if (brightnessRange < 0)
            throw new ArgumentOutOfRangeException(nameof(brightnessRange), "Range must be non-negative");
        if (contrastRange < 0)
            throw new ArgumentOutOfRangeException(nameof(contrastRange), "Range must be non-negative");
        if (saturationRange < 0)
            throw new ArgumentOutOfRangeException(nameof(saturationRange), "Range must be non-negative");
        if (hueRange < 0 || hueRange > 0.5)
            throw new ArgumentOutOfRangeException(nameof(hueRange), "Hue range must be between 0 and 0.5");

        BrightnessRange = brightnessRange;
        ContrastRange = contrastRange;
        SaturationRange = saturationRange;
        HueRange = hueRange;
    }

    /// <summary>
    /// Applies the color jitter to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Only apply to color images
        if (channels < 3)
        {
            return result;
        }

        // Sample random adjustment factors
        double brightnessFactor = 1.0;
        double contrastFactor = 1.0;
        double saturationFactor = 1.0;
        double hueShift = 0.0;

        if (BrightnessRange > 0)
        {
            brightnessFactor = context.GetRandomDouble(1 - BrightnessRange, 1 + BrightnessRange);
        }
        if (ContrastRange > 0)
        {
            contrastFactor = context.GetRandomDouble(1 - ContrastRange, 1 + ContrastRange);
        }
        if (SaturationRange > 0)
        {
            saturationFactor = context.GetRandomDouble(1 - SaturationRange, 1 + SaturationRange);
        }
        if (HueRange > 0)
        {
            hueShift = context.GetRandomDouble(-HueRange, HueRange);
        }

        // Randomize the order of operations for more variety
        int[] order = [0, 1, 2, 3];
        ShuffleArray(order, context);

        // Calculate mean for contrast adjustment
        double[] channelMeans = new double[3];
        int totalPixels = height * width;

        for (int c = 0; c < 3; c++)
        {
            double sum = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    sum += Convert.ToDouble(data.GetPixel(y, x, c));
                }
            }
            channelMeans[c] = sum / totalPixels;
        }

        // Value range
        double minValue = data.IsNormalized ? 0.0 : 0.0;
        double maxValue = data.IsNormalized ? 1.0 : 255.0;

        // Luminance weights
        const double rWeight = 0.299;
        const double gWeight = 0.587;
        const double bWeight = 0.114;

        // Apply all adjustments per pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double r = Convert.ToDouble(data.GetPixel(y, x, 0));
                double g = Convert.ToDouble(data.GetPixel(y, x, 1));
                double b = Convert.ToDouble(data.GetPixel(y, x, 2));

                // Apply operations in random order
                foreach (int op in order)
                {
                    switch (op)
                    {
                        case 0: // Brightness
                            r *= brightnessFactor;
                            g *= brightnessFactor;
                            b *= brightnessFactor;
                            break;

                        case 1: // Contrast
                            r = channelMeans[0] + (r - channelMeans[0]) * contrastFactor;
                            g = channelMeans[1] + (g - channelMeans[1]) * contrastFactor;
                            b = channelMeans[2] + (b - channelMeans[2]) * contrastFactor;
                            break;

                        case 2: // Saturation
                            double gray = rWeight * r + gWeight * g + bWeight * b;
                            r = gray + (r - gray) * saturationFactor;
                            g = gray + (g - gray) * saturationFactor;
                            b = gray + (b - gray) * saturationFactor;
                            break;

                        case 3: // Hue
                            if (Math.Abs(hueShift) > 0.001)
                            {
                                // Convert to HSV, shift hue, convert back
                                RgbToHsv(r, g, b, maxValue, out double h, out double s, out double v);
                                h = (h + hueShift) % 1.0;
                                if (h < 0) h += 1.0;
                                HsvToRgb(h, s, v, maxValue, out r, out g, out b);
                            }
                            break;
                    }
                }

                // Clamp to valid range
                r = Math.Max(minValue, Math.Min(maxValue, r));
                g = Math.Max(minValue, Math.Min(maxValue, g));
                b = Math.Max(minValue, Math.Min(maxValue, b));

                result.SetPixel(y, x, 0, (T)Convert.ChangeType(r, typeof(T)));
                result.SetPixel(y, x, 1, (T)Convert.ChangeType(g, typeof(T)));
                result.SetPixel(y, x, 2, (T)Convert.ChangeType(b, typeof(T)));

                // Keep alpha channel unchanged if present
                for (int c = 3; c < channels; c++)
                {
                    result.SetPixel(y, x, c, data.GetPixel(y, x, c));
                }
            }
        }

        return result;
    }

    private static void ShuffleArray(int[] array, AugmentationContext<T> context)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = context.GetRandomInt(0, i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    private static void RgbToHsv(double r, double g, double b, double maxValue, out double h, out double s, out double v)
    {
        // Normalize to 0-1 range
        r /= maxValue;
        g /= maxValue;
        b /= maxValue;

        double max = Math.Max(r, Math.Max(g, b));
        double min = Math.Min(r, Math.Min(g, b));
        double delta = max - min;

        v = max;
        s = Math.Abs(max) < 1e-10 ? 0 : delta / max;

        if (Math.Abs(delta) < 1e-10)
        {
            h = 0;
        }
        else if (Math.Abs(max - r) < 1e-10)
        {
            h = ((g - b) / delta) % 6;
        }
        else if (Math.Abs(max - g) < 1e-10)
        {
            h = (b - r) / delta + 2;
        }
        else
        {
            h = (r - g) / delta + 4;
        }

        h /= 6;
        if (h < 0) h += 1;
    }

    private static void HsvToRgb(double h, double s, double v, double maxValue, out double r, out double g, out double b)
    {
        double c = v * s;
        double x = c * (1 - Math.Abs((h * 6) % 2 - 1));
        double m = v - c;

        double r1, g1, b1;
        int sector = (int)(h * 6) % 6;

        switch (sector)
        {
            case 0: r1 = c; g1 = x; b1 = 0; break;
            case 1: r1 = x; g1 = c; b1 = 0; break;
            case 2: r1 = 0; g1 = c; b1 = x; break;
            case 3: r1 = 0; g1 = x; b1 = c; break;
            case 4: r1 = x; g1 = 0; b1 = c; break;
            default: r1 = c; g1 = 0; b1 = x; break;
        }

        r = (r1 + m) * maxValue;
        g = (g1 + m) * maxValue;
        b = (b1 + m) * maxValue;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["brightness_range"] = BrightnessRange;
        parameters["contrast_range"] = ContrastRange;
        parameters["saturation_range"] = SaturationRange;
        parameters["hue_range"] = HueRange;
        return parameters;
    }
}
