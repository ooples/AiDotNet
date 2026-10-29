using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety.Image;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Abstract base class for keyed image watermarkers: each adds a pseudo-random pattern derived from a secret key to a
/// transform of the image, and detects it by correlating the same transform against the same pattern.
/// </summary>
/// <remarks>
/// <para>
/// The watermark is computed on the image's luminance (the mean over channels) and the same spatial change is added
/// to every channel. Detection is a correlation test: with the selected transform coefficients c and the key's
/// pattern x, z = sum(c_k * x_k) / sqrt(sum(c_k^2)). Without the watermark, or with a different key, z is
/// approximately standard normal; with it, z grows with the strength and with the number of coefficients.
/// <see cref="DetectWatermark"/> maps z to a confidence: 0 up to z = 3 (a one-sided false-positive rate near 0.1%),
/// rising linearly to 1 at z = 6. <see cref="EvaluateImage"/> reports a finding from <see cref="ReportThreshold"/> (by default confidence 0.3, z = 3.9).
/// </para>
/// <para>
/// The embedded amplitude per coefficient is the strength times the larger of the coefficients' robust spread and 2%
/// of the image's value range, so smooth images still receive a watermark that survives mild noise.
/// </para>
/// <para>
/// <b>For Beginners:</b> the key works like a password. Only someone with the same key can tell the watermark is
/// there. The watermark is spread thinly over many coefficients, so it is invisible in any one place but adds up
/// to a strong signal when you know exactly what pattern to look for.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ImageWatermarkerBase<T> : ImageSafetyModuleBase<T>, IImageWatermarker<T>
{
    /// <summary>The key used when none is given.</summary>
    public const int DefaultKey = 0x5EED;

    /// <summary>The fewest watermark coefficients an image must offer to carry a watermark.</summary>
    protected const int MinimumCoefficients = 64;

    private const double MinimumAmplitudeFraction = 0.02;

    /// <summary>
    /// The watermark strength: the embedded amplitude relative to the spread of the coefficients it is added to.
    /// </summary>
    protected readonly double WatermarkStrength;

    /// <summary>The secret key the watermark pattern is derived from.</summary>
    protected readonly int WatermarkKey;

    /// <summary>
    /// Initializes the image watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Embedding strength relative to the spread of the host coefficients.</param>
    /// <param name="key">The secret key the watermark pattern is derived from.</param>
    protected ImageWatermarkerBase(double watermarkStrength, int key)
    {
        if (double.IsNaN(watermarkStrength) || double.IsInfinity(watermarkStrength) || watermarkStrength <= 0)
            throw new ArgumentOutOfRangeException(nameof(watermarkStrength), "Watermark strength must be a positive, finite number.");
        WatermarkStrength = watermarkStrength;
        WatermarkKey = key;
    }

    /// <summary>The detection confidence from which <see cref="EvaluateImage"/> reports a finding (never at confidence 0). Default: 0.3 (z = 3.9).</summary>
    protected virtual double ReportThreshold => 0.3;

    /// <summary>A short description of the watermark, used in findings (for example "Fourier-ring watermark").</summary>
    protected abstract string WatermarkDescription { get; }

    /// <summary>The number of transform coefficients the watermark uses for an image of this size.</summary>
    protected abstract int CoefficientCount(int height, int width);

    /// <summary>
    /// The spatial change that embeds the watermark in <paramref name="luminance"/>.
    /// </summary>
    /// <param name="luminance">The image's luminance, <c>[height, width]</c>.</param>
    /// <param name="valueScale">The width of the image's value range, used for the minimum amplitude.</param>
    protected abstract double[,] WatermarkDelta(double[,] luminance, double valueScale);

    /// <summary>The correlation z-score of <paramref name="luminance"/> against the key's pattern.</summary>
    protected abstract double WatermarkZ(double[,] luminance);

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The image is too small to carry the watermark.</exception>
    public Tensor<T> EmbedWatermark(Tensor<T> image)
    {
        var (channels, height, width) = ImageDimensions(image);
        int count = CoefficientCount(height, width);
        if (count < MinimumCoefficients)
            throw new ArgumentException(
                $"A {height}x{width} image offers {count} watermark coefficients; at least {MinimumCoefficients} are needed.",
                nameof(image));

        var values = ToDoubles(image);
        var (low, high, nominal) = NominalRange(values);
        double valueScale = nominal ? high - low : StandardDeviation(values);
        if (!(valueScale > 0)) valueScale = 1.0;

        var delta = WatermarkDelta(Luminance(values, channels, height, width), valueScale);
        int plane = height * width;
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < plane; i++)
            {
                double v = values[c * plane + i] + delta[i / width, i % width];
                values[c * plane + i] = nominal ? Math.Max(low, Math.Min(high, v)) : v;
            }
        }

        return FromDoubles(image, values);
    }

    /// <inheritdoc />
    public double DetectWatermark(Tensor<T> image)
    {
        var (channels, height, width) = ImageDimensions(image);
        if (CoefficientCount(height, width) < MinimumCoefficients) return 0;
        double z = WatermarkZ(Luminance(ToDoubles(image), channels, height, width));
        // Non-finite pixels give no usable correlation, so they are no evidence of a watermark.
        if (double.IsNaN(z) || double.IsInfinity(z)) return 0;
        return Math.Max(0.0, Math.Min(1.0, (z - 3.0) / 3.0));
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        double score = DetectWatermark(image);
        // A confidence of 0 is no evidence at all, so even a zero threshold does not report it.
        if (score > 0 && score >= ReportThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = score,
                Description = $"{WatermarkDescription} detected (confidence: {score:F3}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>The per-coefficient amplitude for coefficients whose spread is <paramref name="coefficientSpread"/>.</summary>
    protected double Amplitude(double coefficientSpread, double valueScale) =>
        WatermarkStrength * Math.Max(coefficientSpread, MinimumAmplitudeFraction * valueScale);

    /// <summary>
    /// The key's pseudo-random +/-1 pattern of length <paramref name="count"/>. <paramref name="salt"/> separates the
    /// patterns of different watermarkers that share a key.
    /// </summary>
    protected double[] KeyPattern(int count, int salt)
    {
        var random = RandomHelper.CreateSeededRandom(unchecked(WatermarkKey * 31 + salt));
        var pattern = new double[count];
        for (int i = 0; i < count; i++) pattern[i] = random.Next(2) == 0 ? -1.0 : 1.0;
        return pattern;
    }

    /// <summary>
    /// The correlation z-score of <paramref name="coefficients"/> against <paramref name="pattern"/>:
    /// sum(c * x) / sqrt(sum(c^2)), approximately standard normal when the pattern is absent, and 0 when the coefficients
    /// are at rounding-error level.
    /// </summary>
    protected static double CorrelationZ(double[] coefficients, double[] pattern)
    {
        double dot = 0, energy = 0;
        for (int i = 0; i < coefficients.Length; i++)
        {
            dot += coefficients[i] * pattern[i];
            energy += coefficients[i] * coefficients[i];
        }

        // Coefficients at rounding-error level (an RMS below 1e-9) carry no evidence either way.
        return energy > 1e-18 * coefficients.Length ? dot / Math.Sqrt(energy) : 0;
    }

    /// <summary>
    /// A robust spread of zero-mean coefficients: 1.4826 times the median absolute value, which equals the standard
    /// deviation for Gaussian coefficients but is not inflated by the few strong coefficients of edges and textures.
    /// </summary>
    protected static double RobustSpread(double[] coefficients)
    {
        if (coefficients.Length == 0) return 0;
        var magnitudes = new double[coefficients.Length];
        for (int i = 0; i < coefficients.Length; i++) magnitudes[i] = Math.Abs(coefficients[i]);
        Array.Sort(magnitudes);
        int mid = magnitudes.Length / 2;
        double median = magnitudes.Length % 2 == 1 ? magnitudes[mid] : 0.5 * (magnitudes[mid - 1] + magnitudes[mid]);
        return 1.4826 * median;
    }

    /// <summary>The sample standard deviation of <paramref name="values"/> (0 for fewer than two values).</summary>
    protected static double StandardDeviation(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = 0;
        foreach (double v in values) mean += v;
        mean /= values.Length;
        double sum = 0;
        foreach (double v in values) sum += (v - mean) * (v - mean);
        return Math.Sqrt(sum / (values.Length - 1));
    }

    /// <summary>The matrix product <paramref name="left"/> * <paramref name="x"/> * transpose(<paramref name="right"/>).</summary>
    protected static double[,] SeparableProduct(double[,] left, double[,] x, double[,] right)
    {
        int p = left.GetLength(0), h = x.GetLength(0), w = x.GetLength(1), q = right.GetLength(0);
        var rows = new double[p, w];
        for (int i = 0; i < p; i++)
            for (int k = 0; k < h; k++)
            {
                double a = left[i, k];
                if (a == 0) continue;
                for (int j = 0; j < w; j++) rows[i, j] += a * x[k, j];
            }

        var result = new double[p, q];
        for (int i = 0; i < p; i++)
            for (int j = 0; j < q; j++)
            {
                double sum = 0;
                for (int k = 0; k < w; k++) sum += rows[i, k] * right[j, k];
                result[i, j] = sum;
            }

        return result;
    }

    /// <summary>The transpose of <paramref name="matrix"/>.</summary>
    protected static double[,] Transpose(double[,] matrix)
    {
        int r = matrix.GetLength(0), c = matrix.GetLength(1);
        var t = new double[c, r];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) t[j, i] = matrix[i, j];
        return t;
    }

    private static (int Channels, int Height, int Width) ImageDimensions(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var s = image.Shape;
        return s.Length switch
        {
            2 => (1, s[0], s[1]),
            3 => (s[0], s[1], s[2]),
            4 when s[0] == 1 => (s[1], s[2], s[3]),
            _ => throw new ArgumentException(
                $"Expected an image of shape [H, W], [C, H, W] or [1, C, H, W]; got [{string.Join(", ", s.ToArray())}].",
                nameof(image)),
        };
    }

    private static double[,] Luminance(double[] values, int channels, int height, int width)
    {
        var luminance = new double[height, width];
        int plane = height * width;
        for (int c = 0; c < channels; c++)
            for (int i = 0; i < plane; i++) luminance[i / width, i % width] += values[c * plane + i];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) luminance[y, x] /= channels;
        return luminance;
    }

    // Pixel values that all lie in [0, 1] or [0, 255] are kept in that range after embedding.
    private static (double Low, double High, bool Nominal) NominalRange(double[] values)
    {
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        foreach (double v in values)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        if (min >= 0 && max <= 1) return (0, 1, true);
        if (min >= 0 && max <= 255) return (0, 255, true);
        return (min, max, false);
    }

    private double[] ToDoubles(Tensor<T> image)
    {
        var data = image.ToArray();
        var values = new double[data.Length];
        for (int i = 0; i < data.Length; i++) values[i] = NumOps.ToDouble(data[i]);
        return values;
    }

    private Tensor<T> FromDoubles(Tensor<T> image, double[] values)
    {
        var data = new T[values.Length];
        for (int i = 0; i < values.Length; i++) data[i] = NumOps.FromDouble(values[i]);
        return new Tensor<T>(data, image.Shape.ToArray());
    }
}