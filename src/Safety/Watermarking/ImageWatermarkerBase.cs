using System.Security.Cryptography;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety.Image;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Abstract base class for keyed image watermarkers: each adds a pseudo-random pattern derived from a secret key to a
/// transform of the image, and detects it by correlating the same transform against the same pattern.
/// </summary>
/// <remarks>
/// <para>
/// The watermark is computed on the image's luminance (the mean over channels) and the same spatial change is added
/// to every channel. The image is split into tiles of at most <see cref="MaxTileSize"/> pixels per side, and each tile
/// carries its own pattern, so the cost grows linearly with the pixel count and averaging tiles reveals nothing.
/// Images above <see cref="MaxPixels"/> pixels are rejected.
/// </para>
/// <para>
/// Detection is a correlation test over every tile: with the selected transform coefficients c and the key's pattern
/// x, z = sum(c_k * x_k) / sqrt(sum(c_k^2)). Without the watermark, or with a different key, z is approximately
/// standard normal; with it, z grows with the strength and with the number of coefficients.
/// <see cref="DetectWatermark"/> maps z to a confidence: 0 up to z = 3 (a one-sided false-positive rate near 0.1%),
/// rising linearly to 1 at z = 6. <see cref="EvaluateImage"/> reports a finding from <see cref="ReportThreshold"/>
/// (by default confidence 0.3, z = 3.9).
/// </para>
/// <para>
/// The pattern is derived with HMAC-SHA256 from the key, so the key is the whole secret. The default,
/// <see cref="PublicKey"/>, is published here: anyone can reproduce, and therefore forge, a mark made with it. Use a
/// long random key of your own (for example 32 random bytes in base64) to mark content as yours.
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
    /// <summary>
    /// The key used when none is given. It is public, so marks made with it prove nothing about who made them.
    /// </summary>
    public const string PublicKey = "aidotnet-public-image-watermark-key";

    /// <summary>The largest tile side, in pixels; larger images are watermarked tile by tile.</summary>
    public const int MaxTileSize = 64;

    /// <summary>The fewest watermark coefficients each tile must offer to carry a watermark.</summary>
    protected const int MinimumCoefficients = 64;

    private const double MinimumAmplitudeFraction = 0.02;

    private readonly byte[] _keyBytes;

    /// <summary>
    /// The watermark strength: the embedded amplitude relative to the spread of the coefficients it is added to.
    /// </summary>
    protected readonly double WatermarkStrength;

    /// <summary>
    /// Initializes the image watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Embedding strength relative to the spread of the host coefficients.</param>
    /// <param name="key">The secret key the watermark pattern is derived from; <c>null</c> uses <see cref="PublicKey"/>.</param>
    protected ImageWatermarkerBase(double watermarkStrength, string? key)
    {
        if (double.IsNaN(watermarkStrength) || double.IsInfinity(watermarkStrength) || watermarkStrength <= 0)
            throw new ArgumentOutOfRangeException(nameof(watermarkStrength), "Watermark strength must be a positive, finite number.");
        if (key is not null && key.Length == 0)
            throw new ArgumentException("A watermark key must not be empty; pass null for the public key.", nameof(key));
        WatermarkStrength = watermarkStrength;
        _keyBytes = Encoding.UTF8.GetBytes(key ?? PublicKey);
    }

    /// <summary>
    /// The largest image, in pixels per channel, the watermarker accepts. Default: 2048 x 2048. The cost is linear in
    /// the pixel count: measured on one core, detecting in a 1024 x 1024 RGB image takes about 0.9 s (DCT) and 1.7 s
    /// (Fourier), and embedding about twice that.
    /// </summary>
    protected virtual long MaxPixels => 2048L * 2048L;

    /// <summary>The detection confidence from which <see cref="EvaluateImage"/> reports a finding (never at confidence 0). Default: 0.3 (z = 3.9).</summary>
    protected virtual double ReportThreshold => 0.3;

    /// <summary>A short description of the watermark, used in findings (for example "Fourier-ring watermark").</summary>
    protected abstract string WatermarkDescription { get; }

    /// <summary>Separates this watermarker's patterns from those of other watermarkers that share a key.</summary>
    protected abstract int PatternSalt { get; }

    /// <summary>The number of transform coefficients the watermark uses in a tile of this size.</summary>
    protected abstract int CoefficientCount(int height, int width);

    /// <summary>The watermark coefficients of <paramref name="tile"/>, <see cref="CoefficientCount"/> of them, in a fixed order.</summary>
    protected abstract double[] TileCoefficients(double[,] tile);

    /// <summary>The spatial change that adds <paramref name="coefficientChanges"/> to a tile's watermark coefficients.</summary>
    protected abstract double[,] TileDelta(int height, int width, double[] coefficientChanges);

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The image is too small to carry the watermark, or larger than <see cref="MaxPixels"/>.</exception>
    public Tensor<T> EmbedWatermark(Tensor<T> image)
    {
        var (channels, height, width) = ImageDimensions(image);
        int smallest = SmallestTileCoefficients(height, width);
        if (smallest < MinimumCoefficients)
            throw new ArgumentException(
                $"A {height}x{width} image offers {smallest} watermark coefficients per tile; at least {MinimumCoefficients} are needed.",
                nameof(image));

        var values = ToDoubles(image);
        var (low, high, nominal) = NominalRange(values);
        double valueScale = nominal ? high - low : StandardDeviation(values);
        if (!(valueScale > 0)) valueScale = 1.0;

        var luminance = Luminance(values, channels, height, width);
        var delta = new double[height, width];
        int tileIndex = 0;
        foreach (var (y0, x0, h, w) in Tiles(height, width))
        {
            var coefficients = TileCoefficients(Crop(luminance, y0, x0, h, w));
            double amplitude = Amplitude(RobustSpread(coefficients), valueScale);
            var pattern = KeyPattern(coefficients.Length, tileIndex++);
            var changes = new double[pattern.Length];
            for (int k = 0; k < changes.Length; k++) changes[k] = amplitude * pattern[k];
            var tileDelta = TileDelta(h, w, changes);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) delta[y0 + y, x0 + x] = tileDelta[y, x];
        }

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
    /// <exception cref="ArgumentException">The image is larger than <see cref="MaxPixels"/>.</exception>
    public double DetectWatermark(Tensor<T> image)
    {
        var (channels, height, width) = ImageDimensions(image);
        if (SmallestTileCoefficients(height, width) < MinimumCoefficients) return 0;

        var luminance = Luminance(ToDoubles(image), channels, height, width);
        double dot = 0, energy = 0;
        int count = 0, tileIndex = 0;
        foreach (var (y0, x0, h, w) in Tiles(height, width))
        {
            var coefficients = TileCoefficients(Crop(luminance, y0, x0, h, w));
            var pattern = KeyPattern(coefficients.Length, tileIndex++);
            for (int k = 0; k < coefficients.Length; k++)
            {
                dot += coefficients[k] * pattern[k];
                energy += coefficients[k] * coefficients[k];
            }

            count += coefficients.Length;
        }

        // Coefficients at rounding-error level (an RMS below 1e-9) carry no evidence either way.
        double z = energy > 1e-18 * count ? dot / Math.Sqrt(energy) : 0;
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

    private double Amplitude(double coefficientSpread, double valueScale) =>
        WatermarkStrength * Math.Max(coefficientSpread, MinimumAmplitudeFraction * valueScale);

    // The key's +/-1 pattern for one tile: the bits of HMAC-SHA256(key, salt || tile || block), 256 per block.
    private double[] KeyPattern(int count, int tileIndex)
    {
        var pattern = new double[count];
        var message = new byte[12];
        WriteInt32(message, 0, PatternSalt);
        WriteInt32(message, 4, tileIndex);
        using var hmac = new HMACSHA256(_keyBytes);
        for (int block = 0; block * 256 < count; block++)
        {
            WriteInt32(message, 8, block);
            var digest = hmac.ComputeHash(message);
            for (int bit = 0; bit < 256 && block * 256 + bit < count; bit++)
                pattern[block * 256 + bit] = ((digest[bit >> 3] >> (bit & 7)) & 1) == 0 ? -1.0 : 1.0;
        }

        return pattern;
    }

    private static void WriteInt32(byte[] buffer, int offset, int value)
    {
        buffer[offset] = (byte)value;
        buffer[offset + 1] = (byte)(value >> 8);
        buffer[offset + 2] = (byte)(value >> 16);
        buffer[offset + 3] = (byte)(value >> 24);
    }

    // 1.4826 times the median absolute value: the standard deviation for Gaussian coefficients, but not inflated by
    // the few strong coefficients of edges and textures.
    private static double RobustSpread(double[] coefficients)
    {
        if (coefficients.Length == 0) return 0;
        var magnitudes = new double[coefficients.Length];
        for (int i = 0; i < coefficients.Length; i++) magnitudes[i] = Math.Abs(coefficients[i]);
        Array.Sort(magnitudes);
        int mid = magnitudes.Length / 2;
        double median = magnitudes.Length % 2 == 1 ? magnitudes[mid] : 0.5 * (magnitudes[mid - 1] + magnitudes[mid]);
        return 1.4826 * median;
    }

    private static double StandardDeviation(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = 0;
        foreach (double v in values) mean += v;
        mean /= values.Length;
        double sum = 0;
        foreach (double v in values) sum += (v - mean) * (v - mean);
        return Math.Sqrt(sum / (values.Length - 1));
    }

    // Near-equal tiles of at most MaxTileSize per side, row-major.
    private static IEnumerable<(int Y0, int X0, int Height, int Width)> Tiles(int height, int width)
    {
        int rows = (height + MaxTileSize - 1) / MaxTileSize, cols = (width + MaxTileSize - 1) / MaxTileSize;
        for (int r = 0; r < rows; r++)
        {
            int y0 = (int)((long)r * height / rows), y1 = (int)((long)(r + 1) * height / rows);
            for (int c = 0; c < cols; c++)
            {
                int x0 = (int)((long)c * width / cols), x1 = (int)((long)(c + 1) * width / cols);
                yield return (y0, x0, y1 - y0, x1 - x0);
            }
        }
    }

    private int SmallestTileCoefficients(int height, int width)
    {
        int rows = (height + MaxTileSize - 1) / MaxTileSize, cols = (width + MaxTileSize - 1) / MaxTileSize;
        return CoefficientCount(height / rows, width / cols);
    }

    private static double[,] Crop(double[,] source, int y0, int x0, int height, int width)
    {
        var tile = new double[height, width];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) tile[y, x] = source[y0 + y, x0 + x];
        return tile;
    }

    private (int Channels, int Height, int Width) ImageDimensions(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var s = image.Shape;
        var (channels, height, width) = s.Length switch
        {
            2 => (1, s[0], s[1]),
            3 => (s[0], s[1], s[2]),
            4 when s[0] == 1 => (s[1], s[2], s[3]),
            _ => throw new ArgumentException(
                $"Expected an image of shape [H, W], [C, H, W] or [1, C, H, W]; got [{string.Join(", ", s.ToArray())}].",
                nameof(image)),
        };
        if ((long)height * width > MaxPixels)
            throw new ArgumentException(
                $"A {height}x{width} image exceeds the {MaxPixels}-pixel limit of {ModuleName}.", nameof(image));
        return (channels, height, width);
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