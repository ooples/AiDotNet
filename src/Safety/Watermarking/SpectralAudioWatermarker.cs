using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Keyed patchwork audio watermark: the key splits each frame's 0.5-6 kHz block-DCT coefficients into two halves,
/// raises the magnitudes of one and lowers those of the other, and detection tests the log-magnitude difference.
/// </summary>
/// <remarks>
/// <para>
/// This is the patchwork scheme (Bender et al. 1996) in the DCT domain, as in the modified patchwork algorithm of
/// Yeo and Kim (2003), with a keyed split per frame. Embedding multiplies every band coefficient by
/// exp(beta x_k), beta = 0.3 x strength, where x_k is the key's +/-1 pattern: a change of about +/-1.3 dB at the
/// default strength. Unlike additive spread spectrum, the change is proportional to each coefficient, so it never adds
/// energy where the audio has none.
/// </para>
/// <para>
/// Detection takes the log-magnitudes of the band (floored at 1e-3 of the band RMS), standardises them by their median
/// and robust spread, clips them to +/-3, centres them, and correlates with the key's pattern: the patchwork statistic
/// "mean of half A minus mean of half B" with a variance that holds for any audio, so z is near N(0, 1) without the
/// mark.
/// </para>
/// <para>
/// <b>For Beginners:</b> the key picks half of the frequencies in every short slice of audio; the watermark makes
/// those slightly louder and the others slightly quieter. Nobody can hear a change this small, but with the key you
/// can measure that "your" half is consistently louder.
/// </para>
/// <para>
/// <b>References:</b>
/// - Bender, Gruhl, Morimoto, Lu. Techniques for Data Hiding (IBM Systems Journal, 1996)
/// - Yeo, Kim. Modified Patchwork Algorithm: A Novel Audio Watermarking Scheme (IEEE TSAP, 2003)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Modified Patchwork Algorithm: A Novel Audio Watermarking Scheme",
    "https://doi.org/10.1109/TSA.2003.812145",
    Year = 2003,
    Authors = "In-Kwon Yeo, Hyoung Joong Kim")]
public class SpectralAudioWatermarker<T> : AudioWatermarkerBase<T>
{
    /// <inheritdoc />
    public override string ModuleName => "SpectralAudioWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Patchwork audio watermark";

    /// <inheritdoc />
    protected override int PatternSalt => 0x50415431; // "PAT1"

    /// <inheritdoc />
    protected override double BandLowHz => 500.0;

    /// <inheritdoc />
    protected override double BandHighHz => 6000.0;

    /// <summary>
    /// Initializes a new patchwork audio watermarker.
    /// </summary>
    /// <param name="watermarkStrength">
    /// Embedding strength, positive: magnitudes are scaled by exp(+/-0.3 x strength). Default: 0.5 (about +/-1.3 dB).
    /// </param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    public SpectralAudioWatermarker(double watermarkStrength = 0.5, string? key = null) : base(watermarkStrength, key) { }

    /// <inheritdoc />
    protected override void EmbedFrame(double[] band, double[] pattern, double frameRms)
    {
        double beta = 0.3 * WatermarkStrength;
        double up = Math.Exp(beta), down = Math.Exp(-beta);
        for (int k = 0; k < band.Length; k++) band[k] *= pattern[k] > 0 ? up : down;
    }

    /// <inheritdoc />
    protected override bool DetectionWeights(double[] band, double[] weights)
    {
        double sum = 0;
        foreach (double c in band) sum += c * c;
        double rms = Math.Sqrt(sum / band.Length);
        if (!(rms > 1e-12)) return false;

        double floor = 1e-3 * rms;
        var logs = new double[band.Length];
        for (int k = 0; k < band.Length; k++) logs[k] = Math.Log(Math.Max(Math.Abs(band[k]), floor));

        var sorted = (double[])logs.Clone();
        Array.Sort(sorted);
        double median = Median(sorted);
        var deviations = new double[logs.Length];
        for (int k = 0; k < logs.Length; k++) deviations[k] = Math.Abs(logs[k] - median);
        Array.Sort(deviations);
        double spread = 1.4826 * Median(deviations);
        if (!(spread > 1e-12)) return false;

        double mean = 0;
        for (int k = 0; k < logs.Length; k++)
        {
            weights[k] = Math.Max(-3.0, Math.Min(3.0, (logs[k] - median) / spread));
            mean += weights[k];
        }

        mean /= logs.Length;
        for (int k = 0; k < logs.Length; k++) weights[k] -= mean;
        return true;
    }

    private static double Median(double[] sorted)
    {
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
    }
}
