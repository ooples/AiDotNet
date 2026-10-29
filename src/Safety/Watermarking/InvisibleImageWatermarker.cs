using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Keyed spread-spectrum image watermarker that hides a pseudo-random +/-1 sequence in the image's mid- and
/// high-frequency DCT coefficients and detects it by correlation.
/// </summary>
/// <remarks>
/// <para>
/// Follows the spread-spectrum scheme of Cox, Kilian, Leighton and Shamoon (1997): a key-derived sequence is spread
/// over many full-frame DCT coefficients, so each changes imperceptibly. This implementation is the blind (non-informed)
/// variant: the watermark is added (c' = c + a * x) rather than multiplied, so detection needs only the key and
/// not the original image. The coefficients used are those at normalized frequency radius 0.1-0.8, excluding the
/// first row and column, which carry the energy of image borders and straight gradients.
/// </para>
/// <para>
/// <b>For Beginners:</b> the image is rewritten as a sum of wave patterns (the DCT). A secret, random-looking
/// pattern of tiny nudges is added to many of those waves at once. No single nudge is visible, but someone with the
/// key can add up all the nudges and see a clear signal; without the key the nudges look like noise.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Secure Spread Spectrum Watermarking for Multimedia",
    "https://doi.org/10.1109/83.650120",
    Year = 1997,
    Authors = "Ingemar J. Cox, Joe Kilian, F. Thomson Leighton, Talal Shamoon")]
public class InvisibleImageWatermarker<T> : ImageWatermarkerBase<T>
{
    private const int PatternSalt = 0x0D57;
    private const double MinRadius = 0.1;
    private const double MaxRadius = 0.8;

    /// <inheritdoc />
    public override string ModuleName => "InvisibleImageWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Spread-spectrum DCT image watermark";

    /// <summary>
    /// Initializes a new spread-spectrum DCT image watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding amplitude relative to the spread of the DCT coefficients. Default: 0.3.</param>
    /// <param name="key">The secret key the watermark sequence is derived from.</param>
    public InvisibleImageWatermarker(double watermarkStrength = 0.3, int key = DefaultKey)
        : base(watermarkStrength, key) { }

    /// <inheritdoc />
    protected override int CoefficientCount(int height, int width) => Band(height, width).Count;

    /// <inheritdoc />
    protected override double[,] WatermarkDelta(double[,] luminance, double valueScale)
    {
        int height = luminance.GetLength(0), width = luminance.GetLength(1);
        var dctH = DctMatrix(height);
        var dctW = DctMatrix(width);
        var spectrum = SeparableProduct(dctH, luminance, dctW);
        var band = Band(height, width);

        var coefficients = new double[band.Count];
        for (int k = 0; k < band.Count; k++) coefficients[k] = spectrum[band[k].U, band[k].V];
        double amplitude = Amplitude(StandardDeviation(coefficients), valueScale);

        var pattern = KeyPattern(band.Count, PatternSalt);
        var mark = new double[height, width];
        for (int k = 0; k < band.Count; k++) mark[band[k].U, band[k].V] = amplitude * pattern[k];
        return SeparableProduct(Transpose(dctH), mark, Transpose(dctW));
    }

    /// <inheritdoc />
    protected override double WatermarkZ(double[,] luminance)
    {
        int height = luminance.GetLength(0), width = luminance.GetLength(1);
        var spectrum = SeparableProduct(DctMatrix(height), luminance, DctMatrix(width));
        var band = Band(height, width);
        var coefficients = new double[band.Count];
        for (int k = 0; k < band.Count; k++) coefficients[k] = spectrum[band[k].U, band[k].V];
        return CorrelationZ(coefficients, KeyPattern(band.Count, PatternSalt));
    }

    private static List<(int U, int V)> Band(int height, int width)
    {
        var band = new List<(int U, int V)>();
        for (int u = 1; u < height; u++)
            for (int v = 1; v < width; v++)
            {
                double fu = (double)u / height, fv = (double)v / width;
                double radius = Math.Sqrt(fu * fu + fv * fv);
                if (radius >= MinRadius && radius <= MaxRadius) band.Add((u, v));
            }

        return band;
    }

    // Orthonormal DCT-II matrix: row k is c_k * cos(pi * (2n + 1) * k / (2N)).
    private static double[,] DctMatrix(int n)
    {
        var m = new double[n, n];
        for (int k = 0; k < n; k++)
        {
            double scale = Math.Sqrt((k == 0 ? 1.0 : 2.0) / n);
            for (int i = 0; i < n; i++) m[k, i] = scale * Math.Cos(Math.PI * (2 * i + 1) * k / (2.0 * n));
        }

        return m;
    }
}