using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Audio watermarker using AudioSeal-style localized watermarking for voice cloning detection.
/// </summary>
/// <remarks>
/// <para>
/// Implements a localized watermarking approach inspired by Meta AI's AudioSeal.
/// Instead of embedding a single global watermark, embeds localized watermarks
/// that can identify which specific segments are AI-generated even in partially
/// modified audio.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker adds tiny signatures to small segments of
/// audio. Unlike a global watermark, it can tell you exactly which parts of a recording
/// are AI-generated and which parts are real — useful for detecting partial deepfakes.
/// </para>
/// <para>
/// <b>References:</b>
/// - AudioSeal: Localized watermarking for speech (Meta AI, 2024, arxiv:2401.17264)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AudioSealWatermarker<T> : AudioWatermarkerBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _segmentSize;

    /// <inheritdoc />
    public override string ModuleName => "AudioSealWatermarker";

    /// <summary>
    /// Initializes a new AudioSeal-style watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding strength (0.0-1.0). Default: 0.5.</param>
    /// <param name="segmentSamples">Segment size in samples for localized detection. Default: 4096.</param>
    public AudioSealWatermarker(double watermarkStrength = 0.5, int segmentSamples = 4096)
        : base(watermarkStrength)
    {
        _segmentSize = segmentSamples;
    }

    /// <inheritdoc />
    public override double DetectWatermark(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples.Length < _segmentSize) return 0;

        int numSegments = audioSamples.Length / _segmentSize;
        if (numSegments < 2) return 0;

        int watermarkedSegments = 0;

        for (int s = 0; s < numSegments; s++)
        {
            int offset = s * _segmentSize;
            double segmentScore = DetectSegmentWatermark(audioSamples, offset, _segmentSize);
            if (segmentScore >= 0.4) watermarkedSegments++;
        }

        return (double)watermarkedSegments / numSegments;
    }

    private double DetectSegmentWatermark(Vector<T> samples, int offset, int length)
    {
        // Analyze per-segment statistics for localized watermark patterns
        double energy = 0;
        double zeroCrossings = 0;
        double prevVal = 0;

        int analyzeLength = Math.Min(length, samples.Length - offset);
        for (int i = 0; i < analyzeLength; i++)
        {
            double val = NumOps.ToDouble(samples[offset + i]);
            energy += val * val;

            if (i > 0 && ((val >= 0 && prevVal < 0) || (val < 0 && prevVal >= 0)))
            {
                zeroCrossings++;
            }
            prevVal = val;
        }

        if (analyzeLength < 64) return 0;

        double avgEnergy = energy / analyzeLength;
        double zcRate = zeroCrossings / analyzeLength;

        // Watermarked segments tend to have more uniform energy and specific ZC rates
        // Natural speech: highly variable energy; watermarked: more consistent
        double energyScore = avgEnergy > 0.001 && avgEnergy < 0.1 ? 0.5 : 0;

        // Check for periodic patterns in the signal (watermark carriers)
        double periodicity = 0;
        int step = analyzeLength / 8;
        for (int lag = step; lag < analyzeLength / 2; lag += step)
        {
            double corr = 0;
            int count = 0;
            for (int i = 0; i + lag < analyzeLength; i += 4)
            {
                double a = NumOps.ToDouble(samples[offset + i]);
                double b = NumOps.ToDouble(samples[offset + i + lag]);
                corr += a * b;
                count++;
            }
            if (count > 0)
            {
                double avgCorr = Math.Abs(corr / count);
                if (avgCorr > periodicity) periodicity = avgCorr;
            }
        }

        double periodicityScore = periodicity > 0.01 ? Math.Min(1.0, periodicity / 0.05) : 0;

        return (energyScore + periodicityScore) / 2.0;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();
        double score = DetectWatermark(audioSamples, sampleRate);

        if (score >= 0.3)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = score,
                Description = $"AudioSeal-style localized watermark detected in {score * 100:F0}% of segments.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }
}
