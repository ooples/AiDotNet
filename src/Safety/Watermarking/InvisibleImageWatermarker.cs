using System.Collections.Concurrent;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Keyed spread-spectrum image watermarker that hides a pseudo-random +/-1 sequence in the image's mid- and
/// high-frequency block DCT coefficients and detects it by correlation.
/// </summary>
/// <remarks>
/// <para>
/// Builds on the spread-spectrum idea of Cox, Kilian, Leighton and Shamoon (1997), spreading a key-derived sequence
/// over many DCT coefficients so each changes imperceptibly, but it is not their scheme. Cox et al. take a full-frame
/// DCT, embed multiplicatively (c' = c * (1 + a * x)) in the perceptually most significant, mostly low-frequency
/// coefficients, and detect by comparing against the original image. This class is a blind additive variant: the
/// image is cut into independent tiles of at most <see cref="ImageWatermarkerBase{T}.MaxTileSize"/> pixels per side,
/// each tile gets its own DCT and its own key-derived sequence, the sequence is added (c' = c + a * x), and detection
/// needs only the key, not the original. The coefficients used in each tile are those at normalized frequency radius
/// 0.1-0.8, excluding the first row and column, which carry the energy of tile borders and straight gradients.
/// </para>
/// <para>
/// <b>For Beginners:</b> the image is rewritten as a sum of wave patterns (the DCT). A secret, random-looking
/// pattern of tiny nudges is added to many of those waves at once. No single nudge is visible, but someone with the
/// key can add up all the nudges and see a clear signal; without the key the nudges look like noise.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.SignalProcessing)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Secure Spread Spectrum Watermarking for Multimedia",
    "https://doi.org/10.1109/83.650120",
    Year = 1997,
    Authors = "Ingemar J. Cox, Joe Kilian, F. Thomson Leighton, Talal Shamoon")]
public class InvisibleImageWatermarker<T> : ImageWatermarkerBase<T>
{
    private const double MinRadius = 0.1;
    private const double MaxRadius = 0.8;

    // Tiles are at most MaxTileSize per side, so at most MaxTileSize matrices are ever cached.
    private static readonly ConcurrentDictionary<int, (double[,] Dct, double[,] Inverse)> DctMatrices = new();

    /// <inheritdoc />
    public override string ModuleName => "InvisibleImageWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Spread-spectrum DCT image watermark";

    /// <inheritdoc />
    protected override int PatternSalt => 0x0D57;

    /// <summary>
    /// Initializes a new spread-spectrum DCT image watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding amplitude relative to the spread of the DCT coefficients. Default: 0.3.</param>
    /// <param name="key">
    /// The secret key the watermark sequence is derived from. <c>null</c> uses the public
    /// <see cref="ImageWatermarkerBase{T}.PublicKey"/>, whose marks anyone can forge.
    /// </param>
    public InvisibleImageWatermarker(double watermarkStrength = 0.3, string? key = null)
        : base(watermarkStrength, key) { }

    /// <inheritdoc />
    protected override int CoefficientCount(int height, int width) => Band(height, width).Count;

    /// <inheritdoc />
    protected override double[] TileCoefficients(double[,] tile)
    {
        int height = tile.GetLength(0), width = tile.GetLength(1);
        var spectrum = SeparableProduct(Matrices(height).Dct, tile, Matrices(width).Dct);
        var band = Band(height, width);
        var coefficients = new double[band.Count];
        for (int k = 0; k < band.Count; k++) coefficients[k] = spectrum[band[k].U, band[k].V];
        return coefficients;
    }

    /// <inheritdoc />
    protected override double[,] TileDelta(int height, int width, double[] coefficientChanges)
    {
        var band = Band(height, width);
        var mark = new double[height, width];
        for (int k = 0; k < band.Count; k++) mark[band[k].U, band[k].V] = coefficientChanges[k];
        return SeparableProduct(Matrices(height).Inverse, mark, Matrices(width).Inverse);
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

    // Orthonormal DCT-II matrix (row k is c_k * cos(pi * (2n + 1) * k / (2N))) and its inverse, the transpose.
    private static (double[,] Dct, double[,] Inverse) Matrices(int n) => DctMatrices.GetOrAdd(n, size =>
    {
        var m = new double[size, size];
        for (int k = 0; k < size; k++)
        {
            double scale = Math.Sqrt((k == 0 ? 1.0 : 2.0) / size);
            for (int i = 0; i < size; i++) m[k, i] = scale * Math.Cos(Math.PI * (2 * i + 1) * k / (2.0 * size));
        }

        return (m, Transpose(m));
    });
}
