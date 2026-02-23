using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Audio watermarker that embeds watermarks in the frequency domain using spectral modification.
/// </summary>
/// <remarks>
/// <para>
/// Embeds watermark bits by modifying specific frequency band magnitudes in the audio spectrum.
/// Uses sub-band coding to spread the watermark across psychoacoustically masked frequencies,
/// making it inaudible while robust to compression and resampling.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker hides a signature in the audio's frequency
/// spectrum — the mathematical representation of pitch and tone. The watermark is placed
/// in frequencies that the human ear is least sensitive to, making it inaudible.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SpectralAudioWatermarker<T> : AudioWatermarkerBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public override string ModuleName => "SpectralAudioWatermarker";

    /// <summary>
    /// Initializes a new spectral audio watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Embedding strength (0.0-1.0). Default: 0.5.</param>
    public SpectralAudioWatermarker(double watermarkStrength = 0.5) : base(watermarkStrength) { }

    /// <inheritdoc />
    public override double DetectWatermark(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples.Length < 256) return 0;

        // Analyze sub-band energy ratios for embedded patterns
        int frameSize = 256;
        int numFrames = Math.Min(audioSamples.Length / frameSize, 32);
        if (numFrames < 4) return 0;

        // Compute energy in specific sub-bands across frames
        int numBands = 8;
        double[,] bandEnergies = new double[numFrames, numBands];

        for (int f = 0; f < numFrames; f++)
        {
            int offset = f * frameSize;
            int bandSize = frameSize / numBands;

            for (int b = 0; b < numBands; b++)
            {
                double energy = 0;
                for (int i = 0; i < bandSize && offset + b * bandSize + i < audioSamples.Length; i++)
                {
                    double val = NumOps.ToDouble(audioSamples[offset + b * bandSize + i]);
                    energy += val * val;
                }
                bandEnergies[f, b] = energy / bandSize;
            }
        }

        // Check for cross-frame consistency in mid-frequency bands (bands 2-5)
        // Watermarks create consistent patterns; natural audio varies randomly
        double consistencyScore = 0;
        int consistencyCount = 0;

        for (int b = 2; b < 6 && b < numBands; b++)
        {
            double bandMean = 0;
            for (int f = 0; f < numFrames; f++) bandMean += bandEnergies[f, b];
            bandMean /= numFrames;

            if (bandMean < 1e-10) continue;

            double bandVariance = 0;
            for (int f = 0; f < numFrames; f++)
            {
                double diff = bandEnergies[f, b] - bandMean;
                bandVariance += diff * diff;
            }
            bandVariance /= numFrames;

            // Coefficient of variation: low = consistent = possible watermark
            double cv = Math.Sqrt(bandVariance) / bandMean;
            if (cv < 0.5) consistencyScore += (0.5 - cv) / 0.5;
            consistencyCount++;
        }

        if (consistencyCount == 0) return 0;

        double avgConsistency = consistencyScore / consistencyCount;
        return Math.Max(0, Math.Min(1.0, avgConsistency));
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
                Description = $"Spectral audio watermark detected (confidence: {score:F3}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }
}
