using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Keyed spread-spectrum audio watermark: a key-derived +/-1 sequence added to the 1-7 kHz block-DCT coefficients at
/// a level that follows each frame's loudness, detected by correlation.
/// </summary>
/// <remarks>
/// <para>
/// This is direct-sequence spread-spectrum watermarking (Cox et al. 1997, applied to audio by Kirovski and Malvar
/// 2003) on the frames described in <see cref="AudioWatermarkerBase{T}"/>. Per frame the amplitude is
/// 0.2 x strength x max(spread, 0.03 x frame RMS), with spread the robust scale of the band's coefficients: at the
/// default strength the mark sits about 20 dB below the band's own content. Detection normalises the received
/// coefficients by their robust spread, clips them to +/-3 and correlates with the key's pattern, which gives a
/// z-score near N(0, 1) on audio without the mark. One second of 16 kHz speech-like audio typically reaches z of
/// about 9, so the mark is found with full confidence; with another key, z stays near 0.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio watermarking hides a faint, password-protected hiss under the sound, shaped to stay
/// below what you can hear. Anyone with the password can measure it, even after noise is added or the audio is
/// re-saved; without the password it looks like ordinary noise.
/// </para>
/// <para>
/// <b>References:</b>
/// - Cox, Kilian, Leighton, Shamoon. Secure Spread Spectrum Watermarking for Multimedia (IEEE TIP, 1997)
/// - Kirovski, Malvar. Spread-Spectrum Watermarking of Audio Signals (IEEE TSP, 2003)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Spread-Spectrum Watermarking of Audio Signals",
    "https://doi.org/10.1109/TSP.2003.809384",
    Year = 2003,
    Authors = "Darko Kirovski, Henrique S. Malvar")]
public class AudioWatermarker<T> : AudioWatermarkerBase<T>
{
    private readonly double _detectionThreshold;

    /// <inheritdoc />
    public override string ModuleName => "AudioWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Spread-spectrum audio watermark";

    /// <inheritdoc />
    protected override int PatternSalt => 0x41535331; // "ASS1"

    /// <summary>
    /// Initializes a new audio watermarker.
    /// </summary>
    /// <param name="detectionThreshold">
    /// The detection confidence (0-1) from which <see cref="AudioWatermarkerBase{T}.EvaluateAudio"/> reports a finding;
    /// 0 reports any evidence (confidence above 0). Default: 0.5 (a correlation z-score of 4.5).
    /// </param>
    /// <param name="watermarkStrength">
    /// Embedding strength, positive: the amplitude is 0.2 x strength times the band's robust spread. Higher values are
    /// more robust but closer to audible. Default: 0.5 (about 20 dB below the band's content).
    /// </param>
    /// <param name="key">
    /// The secret key the watermark is derived from. <c>null</c> uses the published <see cref="WatermarkKey.Public"/>
    /// key, whose marks anyone can forge.
    /// </param>
    public AudioWatermarker(double detectionThreshold = 0.5, double watermarkStrength = 0.5, string? key = null)
        : base(watermarkStrength, key)
    {
        if (double.IsNaN(detectionThreshold) || detectionThreshold < 0 || detectionThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(detectionThreshold), "Detection threshold must be between 0 and 1.");
        _detectionThreshold = detectionThreshold;
    }

    /// <inheritdoc />
    protected override double ReportThreshold => _detectionThreshold;

    /// <inheritdoc />
    protected override void EmbedFrame(double[] band, double[] pattern, double frameRms) => AdditiveEmbed(band, pattern, frameRms);

    /// <inheritdoc />
    protected override bool DetectionWeights(double[] band, double[] weights) => AdditiveWeights(band, weights);
}
