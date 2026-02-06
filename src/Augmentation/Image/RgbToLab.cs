namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an image between RGB and CIE L*a*b* color space.
/// </summary>
/// <remarks>
/// <para>
/// The CIE L*a*b* color space is designed to be perceptually uniform, meaning that equal
/// numerical changes correspond to roughly equal perceived color differences. The conversion
/// goes through XYZ color space as an intermediate step.
/// </para>
/// <para><b>For Beginners:</b> L*a*b* is a special color space where the distance between
/// two colors matches how different they look to the human eye. L* is lightness (0=black,
/// 100=white), a* goes from green (negative) to red (positive), and b* goes from blue
/// (negative) to yellow (positive).</para>
/// <para><b>Channel layout:</b>
/// <list type="bullet">
/// <item><b>L* (Lightness)</b>: [0, 100]</item>
/// <item><b>a*</b>: Approximately [-128, 127] (green to red)</item>
/// <item><b>b*</b>: Approximately [-128, 127] (blue to yellow)</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Color difference calculations (Delta E)</item>
/// <item>Perceptually uniform color augmentation</item>
/// <item>Color transfer between images</item>
/// <item>Image quality assessment</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbToLab<T> : ImageAugmenterBase<T>
{
    // D65 illuminant reference white point
    private const double RefX = 0.95047;
    private const double RefY = 1.00000;
    private const double RefZ = 1.08883;

    /// <summary>
    /// Gets whether to normalize output to [0, 1] range.
    /// When false, L* is in [0, 100], a* and b* in approximately [-128, 127].
    /// When true, all channels are mapped to [0, 1].
    /// </summary>
    public bool NormalizeOutput { get; }

    /// <summary>
    /// Creates a new RGB to L*a*b* conversion.
    /// </summary>
    /// <param name="normalizeOutput">Whether to normalize output to [0, 1]. Default is false.</param>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public RgbToLab(bool normalizeOutput = false, double probability = 1.0) : base(probability)
    {
        NormalizeOutput = normalizeOutput;
    }

    /// <summary>
    /// Converts the image from RGB to L*a*b* color space.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3)
            throw new InvalidOperationException("Image must have at least 3 channels for RGB to LAB conversion.");

        var result = new ImageTensor<T>(data.Height, data.Width, data.Channels, data.ChannelOrder, ColorSpace.LAB)
        {
            IsNormalized = NormalizeOutput,
            OriginalRange = data.OriginalRange
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

                // Step 1: Linear RGB (remove sRGB gamma)
                r = SrgbToLinear(r);
                g = SrgbToLinear(g);
                b = SrgbToLinear(b);

                // Step 2: RGB to XYZ (sRGB D65)
                double xVal = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
                double yVal = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
                double zVal = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

                // Step 3: XYZ to L*a*b*
                double xr = xVal / RefX;
                double yr = yVal / RefY;
                double zr = zVal / RefZ;

                double fx = LabF(xr);
                double fy = LabF(yr);
                double fz = LabF(zr);

                double lStar = 116.0 * fy - 16.0;
                double aStar = 500.0 * (fx - fy);
                double bStar = 200.0 * (fy - fz);

                if (NormalizeOutput)
                {
                    // Shift a*/b* from [-128, 127] to [0, 255] using center offset,
                    // then normalize to [0, 1]
                    const double LabCenterOffset = 128.0;
                    lStar /= 100.0;
                    aStar = (aStar + LabCenterOffset) / 255.0;
                    bStar = (bStar + LabCenterOffset) / 255.0;
                }

                result.SetPixel(y, x, 0, NumOps.FromDouble(lStar));
                result.SetPixel(y, x, 1, NumOps.FromDouble(aStar));
                result.SetPixel(y, x, 2, NumOps.FromDouble(bStar));

                for (int c = 3; c < data.Channels; c++)
                {
                    result.SetPixel(y, x, c, data.GetPixel(y, x, c));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts L*a*b* values back to RGB.
    /// </summary>
    /// <param name="l">L* value [0, 100].</param>
    /// <param name="a">a* value (approximately [-128, 127]).</param>
    /// <param name="b">b* value (approximately [-128, 127]).</param>
    /// <returns>RGB values in [0, 1].</returns>
    public static (double r, double g, double bOut) LabToRgb(double l, double a, double b)
    {
        // Lab to XYZ
        double fy = (l + 16.0) / 116.0;
        double fx = a / 500.0 + fy;
        double fz = fy - b / 200.0;

        double xr = LabFInverse(fx);
        double yr = LabFInverse(fy);
        double zr = LabFInverse(fz);

        double xVal = xr * RefX;
        double yVal = yr * RefY;
        double zVal = zr * RefZ;

        // XYZ to linear RGB
        double rLin = xVal * 3.2404542 + yVal * -1.5371385 + zVal * -0.4985314;
        double gLin = xVal * -0.9692660 + yVal * 1.8760108 + zVal * 0.0415560;
        double bLin = xVal * 0.0556434 + yVal * -0.2040259 + zVal * 1.0572252;

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

    private static double LabF(double t)
    {
        const double delta = 6.0 / 29.0;
        return t > delta * delta * delta
            ? Math.Pow(t, 1.0 / 3.0)
            : t / (3.0 * delta * delta) + 4.0 / 29.0;
    }

    private static double LabFInverse(double t)
    {
        const double delta = 6.0 / 29.0;
        return t > delta
            ? t * t * t
            : 3.0 * delta * delta * (t - 4.0 / 29.0);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["normalize_output"] = NormalizeOutput;
        return parameters;
    }
}
