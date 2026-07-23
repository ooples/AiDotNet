namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an image between RGB and CIE XYZ color space.
/// </summary>
/// <remarks>
/// <para>
/// CIE XYZ is a device-independent color space that serves as the foundation for most
/// other color space conversions. It was defined by the International Commission on
/// Illumination (CIE) in 1931 based on human color perception experiments.
/// </para>
/// <para><b>For Beginners:</b> XYZ is a "master" color space that represents all colors
/// the human eye can see. Other color spaces (like RGB, LAB) are derived from XYZ.
/// Y represents luminance (brightness), while X and Z represent chromaticity.
/// You typically don't use XYZ directly for augmentation, but it's needed as an
/// intermediate step for conversions to LAB and other spaces.</para>
/// <para><b>Channel layout (D65 illuminant):</b>
/// <list type="bullet">
/// <item><b>X</b>: Mix of cone responses, roughly correlates with red</item>
/// <item><b>Y</b>: Luminance (brightness as perceived by human eye)</item>
/// <item><b>Z</b>: Roughly correlates with blue</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>As intermediate step for color space conversions (RGB → XYZ → LAB)</item>
/// <item>Color science calculations and colorimetry</item>
/// <item>White point adaptation</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbToXyz<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Creates a new RGB to XYZ conversion.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public RgbToXyz(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Converts the image from sRGB to CIE XYZ color space using D65 illuminant.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3)
            throw new InvalidOperationException("Image must have at least 3 channels for RGB to XYZ conversion.");

        var result = new ImageTensor<T>(data.Height, data.Width, data.Channels, data.ChannelOrder, ColorSpace.RGB)
        {
            IsNormalized = data.IsNormalized,
            OriginalRange = data.OriginalRange,
            Metadata = new Dictionary<string, object> { ["color_space"] = "XYZ" }
        };

        for (int y = 0; y < data.Height; y++)
        {
            for (int x = 0; x < data.Width; x++)
            {
                double r = NumOps.ToDouble(data.GetPixel(y, x, 0));
                double g = NumOps.ToDouble(data.GetPixel(y, x, 1));
                double b = NumOps.ToDouble(data.GetPixel(y, x, 2));

                if (!data.IsNormalized)
                {
                    r /= 255.0;
                    g /= 255.0;
                    b /= 255.0;
                }

                // Remove sRGB gamma correction (linearize)
                r = SrgbToLinear(r);
                g = SrgbToLinear(g);
                b = SrgbToLinear(b);

                // sRGB to XYZ (D65 illuminant) using the standard matrix
                double xVal = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
                double yVal = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
                double zVal = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

                result.SetPixel(y, x, 0, NumOps.FromDouble(xVal));
                result.SetPixel(y, x, 1, NumOps.FromDouble(yVal));
                result.SetPixel(y, x, 2, NumOps.FromDouble(zVal));

                for (int c = 3; c < data.Channels; c++)
                {
                    result.SetPixel(y, x, c, data.GetPixel(y, x, c));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts XYZ values back to sRGB.
    /// </summary>
    /// <param name="x">X tristimulus value.</param>
    /// <param name="y">Y tristimulus value.</param>
    /// <param name="z">Z tristimulus value.</param>
    /// <returns>RGB values in [0, 1].</returns>
    public static (double r, double g, double b) XyzToRgb(double x, double y, double z)
    {
        // XYZ to linear RGB
        double rLin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
        double gLin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
        double bLin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

        // Apply sRGB gamma
        double rOut = LinearToSrgb(rLin);
        double gOut = LinearToSrgb(gLin);
        double bOut = LinearToSrgb(bLin);

        return (Math.Max(0, Math.Min(1, rOut)),
                Math.Max(0, Math.Min(1, gOut)),
                Math.Max(0, Math.Min(1, bOut)));
    }

    private static double SrgbToLinear(double value)
    {
        return value <= 0.04045 ? value / 12.92 : Math.Pow((value + 0.055) / 1.055, 2.4);
    }

    private static double LinearToSrgb(double value)
    {
        return value <= 0.0031308 ? 12.92 * value : 1.055 * Math.Pow(value, 1.0 / 2.4) - 0.055;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
