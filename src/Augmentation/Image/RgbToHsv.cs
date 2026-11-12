namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an image between RGB and HSV color spaces.
/// </summary>
/// <remarks>
/// <para>
/// HSV (Hue, Saturation, Value) separates color information (hue) from intensity (value)
/// and purity (saturation). This makes it easier to perform color-based operations like
/// adjusting hue or saturation independently.
/// </para>
/// <para><b>For Beginners:</b> RGB describes colors by mixing red, green, and blue light.
/// HSV describes colors by their shade (hue, 0-360 degrees), how vivid they are
/// (saturation, 0-1), and how bright they are (value, 0-1). HSV is more intuitive for
/// color manipulation.</para>
/// <para><b>Channel layout:</b>
/// <list type="bullet">
/// <item><b>H (Hue)</b>: Color angle, normalized to [0, 1] (representing 0-360 degrees)</item>
/// <item><b>S (Saturation)</b>: Color purity, [0, 1] (0 = gray, 1 = pure color)</item>
/// <item><b>V (Value)</b>: Brightness, [0, 1] (0 = black, 1 = brightest)</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Color-based augmentation (adjusting hue/saturation independently)</item>
/// <item>Color-based object detection or segmentation</item>
/// <item>When you need to separate luminance from chrominance</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbToHsv<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Creates a new RGB to HSV conversion.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public RgbToHsv(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Converts the image from RGB to HSV color space.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3)
            throw new InvalidOperationException("Image must have at least 3 channels for RGB to HSV conversion.");

        var result = new ImageTensor<T>(data.Height, data.Width, data.Channels, data.ChannelOrder, ColorSpace.HSV)
        {
            IsNormalized = data.IsNormalized,
            OriginalRange = data.OriginalRange
        };

        var engine = Engine;
        var image = data.ToPlanarTensor();

        // Ensure values are in [0, 1]
        var (r, g, b) = ImageTensor<T>.UnitRgbPlanes(image, data.IsNormalized);
        var max = engine.TensorMax(r, engine.TensorMax(g, b));
        var min = engine.TensorMin(r, engine.TensorMin(g, b));
        var delta = engine.TensorSubtract(max, min);

        var h = ImageTensor<T>.HexconeHue(r, g, b, max, delta);

        // Saturation: 0 for black, otherwise delta / max
        var black = engine.TensorLessThan(max, NumOps.FromDouble(1e-10));
        var s = engine.TensorWhere(black, new Tensor<T>(max._shape), engine.TensorDivide(delta, max));

        // Value is max; alpha is preserved if present
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image, h, s, max));

        return result;
    }

    /// <summary>
    /// Converts HSV values back to RGB.
    /// </summary>
    /// <param name="h">Hue [0, 1].</param>
    /// <param name="s">Saturation [0, 1].</param>
    /// <param name="v">Value [0, 1].</param>
    /// <returns>RGB values in [0, 1].</returns>
    public static (double r, double g, double b) HsvToRgb(double h, double s, double v)
    {
        double hDeg = h * 360.0;
        double c = v * s;
        double hPrime = hDeg / 60.0;
        double xVal = c * (1 - Math.Abs(hPrime % 2 - 1));
        double m = v - c;

        double r, g, b;
        if (hPrime < 1)      { r = c; g = xVal; b = 0; }
        else if (hPrime < 2) { r = xVal; g = c; b = 0; }
        else if (hPrime < 3) { r = 0; g = c; b = xVal; }
        else if (hPrime < 4) { r = 0; g = xVal; b = c; }
        else if (hPrime < 5) { r = xVal; g = 0; b = c; }
        else                 { r = c; g = 0; b = xVal; }

        return (r + m, g + m, b + m);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
