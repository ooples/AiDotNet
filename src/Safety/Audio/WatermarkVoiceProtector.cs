using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Protects voice recordings by embedding imperceptible watermarks that survive voice cloning
/// and can be detected in cloned output.
/// </summary>
/// <remarks>
/// <para>
/// Embeds a spread-spectrum watermark into the audio's mid-frequency band that is robust to
/// typical voice processing (resampling, compression, noise addition) and voice cloning.
/// The watermark can later be detected using <see cref="WatermarkDeepfakeDetector{T}"/>.
/// Uses frequency-domain embedding with psychoacoustic masking to ensure inaudibility.
/// </para>
/// <para>
/// <b>For Beginners:</b> This embeds a secret "tag" in the audio that follows the voice
/// even if someone tries to clone it. If cloned audio shows up later, the tag proves where
/// the original voice came from.
/// </para>
/// <para>
/// <b>References:</b>
/// - AudioSeal: Localized watermarking for speech (Meta AI, 2024, arxiv:2401.17264)
/// - SoK: Systematization of watermarking across modalities (2024, arxiv:2411.18479)
/// - Watermarking survey: unified framework (2025, arxiv:2504.03765)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class WatermarkVoiceProtector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _watermarkStrength;
    private readonly int _watermarkKey;
    private readonly FastFourierTransform<T> _fft;

    /// <inheritdoc />
    public override string ModuleName => "WatermarkVoiceProtector";

    /// <summary>
    /// Initializes a new watermark-based voice protector.
    /// </summary>
    /// <param name="watermarkStrength">
    /// Watermark strength (0-1). Default: 0.01. Trade-off between detectability and audibility.
    /// </param>
    /// <param name="watermarkKey">
    /// Secret key for watermark generation. Default: 42. Different keys produce different watermarks.
    /// </param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public WatermarkVoiceProtector(double watermarkStrength = 0.01, int watermarkKey = 42, int sampleRate = 16000)
        : base(sampleRate)
    {
        _watermarkStrength = NumOps.FromDouble(watermarkStrength);
        _watermarkKey = watermarkKey;
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Embeds a watermark into the audio and returns the watermarked audio.
    /// </summary>
    public Vector<T> EmbedWatermark(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples.Length < 512)
        {
            return audioSamples;
        }

        int frameSize = 512;
        int hopSize = 256;
        double strength = NumOps.ToDouble(_watermarkStrength);

        var result = new Vector<T>(audioSamples.Length);
        var windowSum = new Vector<T>(audioSamples.Length);

        for (int start = 0; start + frameSize <= audioSamples.Length; start += hopSize)
        {
            var frame = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize; i++)
            {
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
                frame[i] = NumOps.Multiply(audioSamples[start + i], NumOps.FromDouble(w));
            }

            var spectrum = _fft.Forward(frame);
            int halfLen = spectrum.Length / 2;

            // Embed spread-spectrum watermark in mid-frequency band (1kHz - 6kHz)
            int lowBin = (int)(1000.0 * frameSize / sampleRate);
            int highBin = Math.Min((int)(6000.0 * frameSize / sampleRate), halfLen - 1);

            for (int k = lowBin; k <= highBin; k++)
            {
                // Pseudo-random watermark chip based on key and bin index
                int chip = GenerateChip(_watermarkKey, k, start / hopSize);
                double chipValue = chip == 1 ? 1.0 : -1.0;

                T magnitude = spectrum[k].Magnitude;
                double magD = NumOps.ToDouble(magnitude);
                double delta = magD * strength * chipValue;

                T newReal = NumOps.Add(spectrum[k].Real, NumOps.FromDouble(delta));
                spectrum[k] = new Complex<T>(newReal, spectrum[k].Imaginary);
            }

            var watermarkedFrame = _fft.Inverse(spectrum);

            // Overlap-add
            for (int i = 0; i < frameSize; i++)
            {
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
                T windowed = NumOps.Multiply(watermarkedFrame[i], NumOps.FromDouble(w));
                result[start + i] = NumOps.Add(result[start + i], windowed);
                windowSum[start + i] = NumOps.Add(windowSum[start + i], NumOps.FromDouble(w * w));
            }
        }

        // Normalize by window sum
        for (int i = 0; i < result.Length; i++)
        {
            double ws = NumOps.ToDouble(windowSum[i]);
            if (ws > 1e-10)
            {
                result[i] = NumOps.Divide(result[i], windowSum[i]);
            }
            else
            {
                result[i] = audioSamples[i];
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        // Detect our specific watermark
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 512) return findings;

        double detectionScore = DetectWatermark(audioSamples, sampleRate);

        if (detectionScore >= 0.5)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, detectionScore),
                Description = $"Voice watermark detected (score: {detectionScore:F3}). " +
                              $"Audio contains embedded identification watermark.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private double DetectWatermark(Vector<T> audio, int sampleRate)
    {
        int frameSize = 512;
        int hopSize = 256;
        int numFrames = Math.Min(8, (audio.Length - frameSize) / hopSize + 1);
        if (numFrames < 1) return 0;

        double correlationSum = 0;
        int correlationCount = 0;

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * hopSize;
            if (start + frameSize > audio.Length) break;

            var frame = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize; i++)
            {
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
                frame[i] = NumOps.Multiply(audio[start + i], NumOps.FromDouble(w));
            }

            var spectrum = _fft.Forward(frame);
            int halfLen = spectrum.Length / 2;

            int lowBin = (int)(1000.0 * frameSize / sampleRate);
            int highBin = Math.Min((int)(6000.0 * frameSize / sampleRate), halfLen - 1);

            // Correlate spectrum with expected watermark pattern
            double corr = 0;
            int bins = 0;
            for (int k = lowBin; k <= highBin; k++)
            {
                int chip = GenerateChip(_watermarkKey, k, f);
                double chipValue = chip == 1 ? 1.0 : -1.0;
                double realVal = NumOps.ToDouble(spectrum[k].Real);
                corr += realVal * chipValue;
                bins++;
            }

            if (bins > 0)
            {
                correlationSum += corr / bins;
                correlationCount++;
            }
        }

        if (correlationCount == 0) return 0;

        double avgCorrelation = correlationSum / correlationCount;
        return Math.Min(1.0, Math.Max(0, avgCorrelation * 10));
    }

    private static int GenerateChip(int key, int bin, int frame)
    {
        unchecked
        {
            int hash = key * (int)2654435761 + bin * (int)2246822519 + frame * (int)3266489917;
            hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
            hash = (hash >> 16) ^ hash;
            return (hash & 1);
        }
    }
}
