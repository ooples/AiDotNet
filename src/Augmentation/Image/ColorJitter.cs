
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
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var r = engine.TensorSliceAxis(image, 0, 0);
        var g = engine.TensorSliceAxis(image, 0, 1);
        var b = engine.TensorSliceAxis(image, 0, 2);
        T meanR = engine.TensorMean(r);
        T meanG = engine.TensorMean(g);
        T meanB = engine.TensorMean(b);

        // Value range
        double minValue = data.IsNormalized ? 0.0 : 0.0;
        double maxValue = data.IsNormalized ? 1.0 : 255.0;

        // Luminance weights
        const double rWeight = 0.299;
        const double gWeight = 0.587;
        const double bWeight = 0.114;

        T brightness = NumOps.FromDouble(brightnessFactor);
        T contrast = NumOps.FromDouble(contrastFactor);
        T saturation = NumOps.FromDouble(saturationFactor);

        // Apply operations in random order to the whole image
        foreach (int op in order)
        {
            switch (op)
            {
                case 0: // Brightness
                    r = engine.TensorMultiplyScalar(r, brightness);
                    g = engine.TensorMultiplyScalar(g, brightness);
                    b = engine.TensorMultiplyScalar(b, brightness);
                    break;

                case 1: // Contrast
                    r = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorSubtractScalar(r, meanR), contrast), meanR);
                    g = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorSubtractScalar(g, meanG), contrast), meanG);
                    b = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorSubtractScalar(b, meanB), contrast), meanB);
                    break;

                case 2: // Saturation
                    var gray = ImageTensor<T>.MixPlanes(r, g, b, rWeight, gWeight, bWeight);
                    r = engine.TensorAdd(gray, engine.TensorMultiplyScalar(engine.TensorSubtract(r, gray), saturation));
                    g = engine.TensorAdd(gray, engine.TensorMultiplyScalar(engine.TensorSubtract(g, gray), saturation));
                    b = engine.TensorAdd(gray, engine.TensorMultiplyScalar(engine.TensorSubtract(b, gray), saturation));
                    break;

                case 3: // Hue
                    if (Math.Abs(hueShift) > 0.001)
                    {
                        // Convert to HSV, shift hue, convert back
                        (r, g, b) = ShiftHue(engine, r, g, b, hueShift, maxValue);
                    }
                    break;
            }
        }

        // Clamp to valid range; alpha and other extra channels are kept unchanged
        T min = NumOps.FromDouble(minValue);
        T max = NumOps.FromDouble(maxValue);
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image,
            engine.TensorClamp(r, min, max), engine.TensorClamp(g, min, max), engine.TensorClamp(b, min, max)));

        return result;
    }

    private static (Tensor<T> R, Tensor<T> G, Tensor<T> B) ShiftHue(
        IEngine engine, Tensor<T> r, Tensor<T> g, Tensor<T> b, double hueShift, double maxValue)
    {
        // Normalize to 0-1 range
        T scale = NumOps.FromDouble(maxValue);
        r = engine.TensorDivideScalar(r, scale);
        g = engine.TensorDivideScalar(g, scale);
        b = engine.TensorDivideScalar(b, scale);

        var max = engine.TensorMax(r, engine.TensorMax(g, b));
        var min = engine.TensorMin(r, engine.TensorMin(g, b));
        var delta = engine.TensorSubtract(max, min);

        var black = engine.TensorLessThan(engine.TensorAbs(max), NumOps.FromDouble(1e-10));
        var s = engine.TensorWhere(black, ImageTensor<T>.Constant(max, 0.0), engine.TensorDivide(delta, max));
        var h = ImageTensor<T>.HexconeHue(r, g, b, max, delta);
        h = ImageTensor<T>.WrapUnitInterval(engine.TensorAddScalar(h, NumOps.FromDouble(hueShift)));

        var (rr, gg, bb) = ImageTensor<T>.HsvToRgbPlanes(h, s, max);
        return (engine.TensorMultiplyScalar(rr, scale), engine.TensorMultiplyScalar(gg, scale), engine.TensorMultiplyScalar(bb, scale));
    }

    private static void ShuffleArray(int[] array, AugmentationContext<T> context)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = context.GetRandomInt(0, i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
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
