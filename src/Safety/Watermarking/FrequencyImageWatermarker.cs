using System.Collections.Concurrent;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Keyed Fourier-ring image watermarker: adds a key-derived pattern to a ring-shaped (annular) band of the image's
/// Fourier spectrum and detects it by correlation.
/// </summary>
/// <remarks>
/// <para>
/// Follows Tree-Ring watermarks (Wen et al., 2023), which write a key into a circular region of a Fourier spectrum.
/// The key occupies the mid-frequency annulus (normalized radius 0.08-0.4), one independent +/-1 value per
/// coefficient, which is Tree-Ring's random-key variant. Tree-Ring's ring-constant variant (one value per ring) is
/// not used: it has one degree of freedom per ring, so two different keys correlate at about 1/sqrt(rings) and a
/// strongly marked image would be "detected" by strangers' keys too. Tree-Ring places its key in the initial noise of
/// a diffusion model and detects it by inverting the sampler; this class applies the key to a finished image's own
/// spectrum instead, so it works on any image without a generator. It is a post-hoc watermark, not Tree-Ring's
/// generation-time one.
/// </para>
/// <para>
/// The key value is added to the real part of each coefficient and of its conjugate partner (-u, -v), so the change
/// to the image is real. Detection correlates one coefficient of each conjugate pair against the key; coefficients on
/// the first row and column are excluded because image borders and straight gradients concentrate energy there.
/// Images larger than <see cref="ImageWatermarkerBase{T}.MaxTileSize"/> pixels per side are marked tile by tile.
/// </para>
/// <para>
/// <b>For Beginners:</b> the Fourier spectrum describes an image as ripples of different sizes and directions. This
/// watermark slightly strengthens or weakens the medium-sized ripples in a secret pattern. The change is invisible,
/// but anyone with the key can check whether the ripples follow the pattern.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust",
    "https://arxiv.org/abs/2305.20030",
    Year = 2023,
    Authors = "Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein")]
public class FrequencyImageWatermarker<T> : ImageWatermarkerBase<T>
{
    private const double MinRadius = 0.08;
    private const double MaxRadius = 0.4;

    // Tiles are at most MaxTileSize per side, so at most MaxTileSize matrix pairs are ever cached.
    private static readonly ConcurrentDictionary<int, (double[,] Cos, double[,] Sin)> FourierMatrixCache = new();

    /// <inheritdoc />
    public override string ModuleName => "FrequencyImageWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Fourier-ring image watermark";

    /// <inheritdoc />
    protected override int PatternSalt => 0x7EE5;

    /// <summary>
    /// Initializes a new Fourier-ring image watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding amplitude relative to the spread of the spectrum's real parts. Default: 0.5.</param>
    /// <param name="key">
    /// The secret key the watermark pattern is derived from. <c>null</c> uses the public
    /// <see cref="ImageWatermarkerBase{T}.PublicKey"/>, whose marks anyone can forge.
    /// </param>
    public FrequencyImageWatermarker(double watermarkStrength = 0.5, string? key = null)
        : base(watermarkStrength, key) { }

    /// <inheritdoc />
    protected override int CoefficientCount(int height, int width) => DetectionBand(height, width).Count;

    /// <inheritdoc />
    protected override double[] TileCoefficients(double[,] tile)
    {
        int height = tile.GetLength(0), width = tile.GetLength(1);
        var real = RealSpectrum(tile);
        var band = DetectionBand(height, width);
        var coefficients = new double[band.Count];
        for (int k = 0; k < band.Count; k++) coefficients[k] = real[band[k].U, band[k].V];
        return coefficients;
    }

    /// <inheritdoc />
    protected override double[,] TileDelta(int height, int width, double[] coefficientChanges)
    {
        var band = DetectionBand(height, width);
        var mark = new double[height, width];
        for (int k = 0; k < band.Count; k++)
        {
            var (u, v) = band[k];
            mark[u, v] = coefficientChanges[k];
            mark[height - u, width - v] = coefficientChanges[k];
        }

        // The pattern is real and conjugate-symmetric, so its inverse unitary DFT is the real part alone.
        return RealSpectrum(mark);
    }

    // Real part of the unitary 2-D DFT: (C X C^T - S X S^T) / sqrt(HW). For a real, symmetric spectrum the same
    // expression is its inverse.
    private static double[,] RealSpectrum(double[,] x)
    {
        int height = x.GetLength(0), width = x.GetLength(1);
        var (cosH, sinH) = FourierMatrices(height);
        var (cosW, sinW) = FourierMatrices(width);
        var cc = SeparableProduct(cosH, x, cosW);
        var ss = SeparableProduct(sinH, x, sinW);
        double norm = 1.0 / Math.Sqrt((double)height * width);
        var result = new double[height, width];
        for (int u = 0; u < height; u++)
            for (int v = 0; v < width; v++) result[u, v] = (cc[u, v] - ss[u, v]) * norm;
        return result;
    }

    private static (double[,] Cos, double[,] Sin) FourierMatrices(int n) => FourierMatrixCache.GetOrAdd(n, size =>
    {
        var cos = new double[size, size];
        var sin = new double[size, size];
        for (int k = 0; k < size; k++)
            for (int i = 0; i < size; i++)
            {
                double angle = 2.0 * Math.PI * ((long)k * i % size) / size;
                cos[k, i] = Math.Cos(angle);
                sin[k, i] = Math.Sin(angle);
            }

        return (cos, sin);
    });

    private static bool InAnnulus(int u, int v, int height, int width)
    {
        double fu = (double)Math.Min(u, height - u) / height;
        double fv = (double)Math.Min(v, width - v) / width;
        double radius = Math.Sqrt(fu * fu + fv * fv);
        return radius >= MinRadius && radius < MaxRadius;
    }

    // One coefficient of each conjugate pair (u, v) ~ (H - u, W - v) inside the annulus, off the axes.
    private static List<(int U, int V)> DetectionBand(int height, int width)
    {
        var band = new List<(int U, int V)>();
        for (int u = 1; u < height; u++)
            for (int v = 1; v < width; v++)
            {
                int mu = height - u, mv = width - v;
                bool representative = u < mu || (u == mu && v <= mv);
                if (representative && InAnnulus(u, v, height, width)) band.Add((u, v));
            }

        return band;
    }
}
