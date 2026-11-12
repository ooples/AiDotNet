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

        var engine = Engine;
        var image = data.ToPlanarTensor();
        var (r, g, b) = ImageTensor<T>.UnitRgbPlanes(image, data.IsNormalized);

        // Step 1: Linear RGB (remove sRGB gamma)
        r = ImageTensor<T>.SrgbToLinear(r);
        g = ImageTensor<T>.SrgbToLinear(g);
        b = ImageTensor<T>.SrgbToLinear(b);

        // Step 2: RGB to XYZ (sRGB D65)
        var xVal = ImageTensor<T>.MixPlanes(r, g, b, 0.4124564, 0.3575761, 0.1804375);
        var yVal = ImageTensor<T>.MixPlanes(r, g, b, 0.2126729, 0.7151522, 0.0721750);
        var zVal = ImageTensor<T>.MixPlanes(r, g, b, 0.0193339, 0.1191920, 0.9503041);

        // Step 3: XYZ to L*a*b*
        var fx = LabF(engine, engine.TensorDivideScalar(xVal, NumOps.FromDouble(RefX)));
        var fy = LabF(engine, engine.TensorDivideScalar(yVal, NumOps.FromDouble(RefY)));
        var fz = LabF(engine, engine.TensorDivideScalar(zVal, NumOps.FromDouble(RefZ)));

        var lStar = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(fy, NumOps.FromDouble(116.0)), NumOps.FromDouble(16.0));
        var aStar = engine.TensorMultiplyScalar(engine.TensorSubtract(fx, fy), NumOps.FromDouble(500.0));
        var bStar = engine.TensorMultiplyScalar(engine.TensorSubtract(fy, fz), NumOps.FromDouble(200.0));

        if (NormalizeOutput)
        {
            // Shift a*/b* from [-128, 127] to [0, 255] using center offset,
            // then normalize to [0, 1]
            const double LabCenterOffset = 128.0;
            lStar = engine.TensorDivideScalar(lStar, NumOps.FromDouble(100.0));
            aStar = engine.TensorDivideScalar(engine.TensorAddScalar(aStar, NumOps.FromDouble(LabCenterOffset)), NumOps.FromDouble(255.0));
            bStar = engine.TensorDivideScalar(engine.TensorAddScalar(bStar, NumOps.FromDouble(LabCenterOffset)), NumOps.FromDouble(255.0));
        }

        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image, lStar, aStar, bStar));

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

    private static double LinearToSrgb(double value)
    {
        return value <= 0.0031308 ? 12.92 * value : 1.055 * Math.Pow(value, 1.0 / 2.4) - 0.055;
    }

    private static Tensor<T> LabF(IEngine engine, Tensor<T> t)
    {
        const double delta = 6.0 / 29.0;
        var cubeRoot = engine.TensorPower(t, NumOps.FromDouble(1.0 / 3.0));
        var linear = engine.TensorAddScalar(
            engine.TensorDivideScalar(t, NumOps.FromDouble(3.0 * delta * delta)), NumOps.FromDouble(4.0 / 29.0));
        return engine.TensorWhere(engine.TensorGreaterThan(t, NumOps.FromDouble(delta * delta * delta)), cubeRoot, linear);
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
