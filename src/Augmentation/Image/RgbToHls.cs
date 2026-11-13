namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an image between RGB and HLS (Hue, Lightness, Saturation) color spaces.
/// </summary>
/// <remarks>
/// <para>
/// HLS (also called HSL) separates color into hue, lightness, and saturation components.
/// Unlike HSV where value represents the maximum channel, lightness represents the average
/// of the maximum and minimum channels, making it more perceptually uniform.
/// </para>
/// <para><b>For Beginners:</b> HLS is similar to HSV but uses "lightness" instead of "value".
/// Lightness = 0 is always black, lightness = 1 is always white, and lightness = 0.5 gives
/// you the purest colors. This matches how we naturally think about light and dark colors.</para>
/// <para><b>Channel layout:</b>
/// <list type="bullet">
/// <item><b>H (Hue)</b>: Color angle, normalized to [0, 1]</item>
/// <item><b>L (Lightness)</b>: [0, 1] (0 = black, 0.5 = pure color, 1 = white)</item>
/// <item><b>S (Saturation)</b>: [0, 1] (0 = gray, 1 = fully saturated)</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Color manipulation where lightness should be independent of saturation</item>
/// <item>CSS-style color adjustments</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbToHls<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Creates a new RGB to HLS conversion.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public RgbToHls(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Converts the image from RGB to HLS color space.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3)
            throw new InvalidOperationException("Image must have at least 3 channels for RGB to HLS conversion.");

        var result = new ImageTensor<T>(data.Height, data.Width, data.Channels, data.ChannelOrder, ColorSpace.HSL)
        {
            IsNormalized = true,  // HLS output is always in [0, 1] regardless of input range
            OriginalRange = data.OriginalRange
        };

        var engine = Engine;
        var image = data.ToPlanarTensor();

        var (r, g, b) = ImageTensor<T>.UnitRgbPlanes(image, data.IsNormalized);
        var max = engine.TensorMax(r, engine.TensorMax(g, b));
        var min = engine.TensorMin(r, engine.TensorMin(g, b));
        var delta = engine.TensorSubtract(max, min);
        var sum = engine.TensorAdd(max, min);

        // Lightness
        var l = engine.TensorDivideScalar(sum, NumOps.FromDouble(2.0));

        // Saturation: 0 when achromatic, delta / (max + min) when l <= 0.5, else delta / (2 - max - min)
        var dark = engine.TensorDivide(delta, sum);
        var light = engine.TensorDivide(delta, engine.ScalarMinusTensor(NumOps.FromDouble(2.0), sum));
        var bright = engine.TensorGreaterThan(l, NumOps.FromDouble(0.5));
        var achromatic = engine.TensorLessThan(delta, NumOps.FromDouble(1e-10));
        var s = engine.TensorWhere(achromatic, new Tensor<T>(max._shape), engine.TensorWhere(bright, light, dark));

        // Hue
        var h = ImageTensor<T>.HexconeHue(r, g, b, max, delta);

        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image, h, l, s));

        return result;
    }

    /// <summary>
    /// Converts HLS values back to RGB.
    /// </summary>
    /// <param name="h">Hue [0, 1].</param>
    /// <param name="l">Lightness [0, 1].</param>
    /// <param name="s">Saturation [0, 1].</param>
    /// <returns>RGB values in [0, 1].</returns>
    public static (double r, double g, double b) HlsToRgb(double h, double l, double s)
    {
        if (s < 1e-10)
            return (l, l, l);

        double q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        double p = 2 * l - q;

        double r = HueToRgb(p, q, h + 1.0 / 3.0);
        double g = HueToRgb(p, q, h);
        double b = HueToRgb(p, q, h - 1.0 / 3.0);

        return (r, g, b);
    }

    private static double HueToRgb(double p, double q, double t)
    {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1.0 / 6.0) return p + (q - p) * 6 * t;
        if (t < 1.0 / 2.0) return q;
        if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6;
        return p;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
