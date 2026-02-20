using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Detects AI-generated audio by looking for the presence or absence of known watermark
/// patterns (e.g., AudioSeal-style localized watermarks).
/// </summary>
/// <remarks>
/// <para>
/// Many responsible AI audio generators embed watermarks into their output. This detector
/// checks for characteristic patterns of known watermarking schemes: spectral energy in
/// specific sub-bands, phase-based encoding, and spread-spectrum signatures. The absence
/// of any camera/microphone artifacts combined with no watermark is also a signal.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some AI companies embed invisible "watermarks" in the audio they
/// generate — like a secret signature proving it's AI-made. This module looks for those
/// signatures. If found, we know it's AI-generated. If we also detect other AI artifacts
/// but no watermark, it might be an attempt to hide AI origin.
/// </para>
/// <para>
/// <b>References:</b>
/// - AudioSeal: Localized watermarking for speech (Meta AI, 2024, arxiv:2401.17264)
/// - SoK: Systematization of watermarking across modalities (2024, arxiv:2411.18479)
/// - Only 38% of AI generators implement adequate watermarking (2025, arxiv:2503.18156)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class WatermarkDeepfakeDetector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;
    private readonly FastFourierTransform<T> _fft;

    /// <inheritdoc />
    public override string ModuleName => "WatermarkDeepfakeDetector";

    /// <summary>
    /// Initializes a new watermark-based deepfake detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public WatermarkDeepfakeDetector(double threshold = 0.5, int sampleRate = 16000)
        : base(sampleRate)
    {
        _threshold = NumOps.FromDouble(threshold);
        _fft = new FastFourierTransform<T>();
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 1024)
        {
            return findings;
        }

        // Check for various watermark signatures
        double spreadSpectrumScore = DetectSpreadSpectrumWatermark(audioSamples, sampleRate);
        double subBandScore = DetectSubBandWatermark(audioSamples, sampleRate);
        double phaseScore = DetectPhaseWatermark(audioSamples, sampleRate);

        // Also check for natural recording artifacts (their absence is suspicious)
        double naturalArtifactScore = DetectNaturalRecordingArtifacts(audioSamples, sampleRate);

        // Combined watermark detection
        double watermarkScore = Math.Max(spreadSpectrumScore, Math.Max(subBandScore, phaseScore));

        if (watermarkScore >= NumOps.ToDouble(_threshold))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, watermarkScore),
                Description = $"Audio watermark detected (score: {watermarkScore:F3}). " +
                              $"Spread-spectrum: {spreadSpectrumScore:F3}, sub-band: {subBandScore:F3}, " +
                              $"phase: {phaseScore:F3}. This indicates AI-generated audio " +
                              $"from a responsible provider.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        // If no watermark but also no natural artifacts → suspicious
        double suspiciousScore = (1.0 - watermarkScore) * (1.0 - naturalArtifactScore);
        if (suspiciousScore >= NumOps.ToDouble(_threshold))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.AIGenerated,
                Severity = SafetySeverity.Medium,
                Confidence = Math.Min(1.0, suspiciousScore),
                Description = $"Audio lacks both watermarks and natural recording artifacts " +
                              $"(score: {suspiciousScore:F3}). This may indicate AI-generated audio " +
                              $"without responsible watermarking.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private double DetectSpreadSpectrumWatermark(Vector<T> audio, int sampleRate)
    {
        // Spread-spectrum watermarks distribute energy uniformly across frequencies
        int frameSize = 512;
        int numFrames = Math.Min(8, audio.Length / frameSize);
        if (numFrames < 2) return 0;

        // Check for unusually uniform mid-band energy across frames
        var midBandFlatness = new List<double>();

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * frameSize;
            var frame = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize && start + i < audio.Length; i++)
            {
                frame[i] = audio[start + i];
            }

            var spectrum = _fft.Forward(frame);
            int halfLen = spectrum.Length / 2;
            int lowBand = halfLen / 4;
            int highBand = halfLen * 3 / 4;

            // Spectral flatness in mid-band
            double logSum = 0, arithSum = 0;
            int count = 0;
            for (int k = lowBand; k < highBand; k++)
            {
                double mag = NumOps.ToDouble(spectrum[k].Magnitude);
                if (mag > 1e-20)
                {
                    logSum += Math.Log(mag);
                    arithSum += mag;
                    count++;
                }
            }

            if (count > 0 && arithSum > 1e-20)
            {
                double geoMean = Math.Exp(logSum / count);
                double arithMean = arithSum / count;
                midBandFlatness.Add(geoMean / arithMean);
            }
        }

        if (midBandFlatness.Count < 2) return 0;

        // High and consistent mid-band flatness = spread-spectrum watermark
        double meanFlatness = 0;
        foreach (var f in midBandFlatness) meanFlatness += f;
        meanFlatness /= midBandFlatness.Count;

        double variance = 0;
        foreach (var f in midBandFlatness)
        {
            double diff = f - meanFlatness;
            variance += diff * diff;
        }
        variance /= midBandFlatness.Count;

        // Watermarked: mean flatness > 0.6, low variance
        double flatnessScore = Math.Max(0, (meanFlatness - 0.5) * 2);
        double consistencyScore = Math.Max(0, 1.0 - variance * 50);

        return Math.Min(1.0, flatnessScore * consistencyScore);
    }

    private double DetectSubBandWatermark(Vector<T> audio, int sampleRate)
    {
        // Sub-band watermarks embed energy in specific narrow frequency bands
        int frameSize = 1024;
        if (audio.Length < frameSize) return 0;

        var frame = new Vector<T>(frameSize);
        int mid = (audio.Length - frameSize) / 2;
        for (int i = 0; i < frameSize; i++)
        {
            frame[i] = audio[mid + i];
        }

        var spectrum = _fft.Forward(frame);
        int halfLen = spectrum.Length / 2;

        // Compute magnitude spectrum
        var magnitudes = new double[halfLen];
        for (int k = 0; k < halfLen; k++)
        {
            magnitudes[k] = NumOps.ToDouble(spectrum[k].Magnitude);
        }

        // Look for unusually regular energy peaks at specific intervals
        // (watermarks often use fixed frequency spacing)
        int maxPeakCount = 0;
        for (int spacing = 4; spacing <= 32; spacing++)
        {
            int peakCount = 0;
            for (int k = spacing; k < halfLen - spacing; k += spacing)
            {
                bool isPeak = magnitudes[k] > magnitudes[k - 1] &&
                             magnitudes[k] > magnitudes[k + 1] &&
                             magnitudes[k] > magnitudes[k - 2] * 1.5;
                if (isPeak) peakCount++;
            }
            if (peakCount > maxPeakCount) maxPeakCount = peakCount;
        }

        int expectedPeaks = halfLen / 16; // Expected for watermarked audio
        return Math.Min(1.0, (double)maxPeakCount / expectedPeaks);
    }

    private double DetectPhaseWatermark(Vector<T> audio, int sampleRate)
    {
        // Phase-based watermarks encode data in phase differences between frames
        int frameSize = 512;
        int numFrames = Math.Min(4, audio.Length / frameSize);
        if (numFrames < 2) return 0;

        var prevPhases = new double[0];
        var phaseDiffConsistency = new List<double>();

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * frameSize;
            var frame = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize && start + i < audio.Length; i++)
            {
                frame[i] = audio[start + i];
            }

            var spectrum = _fft.Forward(frame);
            int halfLen = spectrum.Length / 2;

            var phases = new double[halfLen];
            for (int k = 0; k < halfLen; k++)
            {
                phases[k] = NumOps.ToDouble(spectrum[k].Phase);
            }

            if (prevPhases.Length > 0)
            {
                // Check if phase differences cluster around discrete values
                // (watermarks encode bits as specific phase offsets)
                int discreteCount = 0;
                for (int k = 10; k < halfLen; k++)
                {
                    double diff = phases[k] - prevPhases[k];
                    // Normalize to [-pi, pi]
                    while (diff > Math.PI) diff -= 2 * Math.PI;
                    while (diff < -Math.PI) diff += 2 * Math.PI;

                    // Check if close to 0, pi/2, pi, or -pi/2
                    double absDiff = Math.Abs(diff);
                    if (absDiff < 0.1 || Math.Abs(absDiff - Math.PI / 2) < 0.1 ||
                        Math.Abs(absDiff - Math.PI) < 0.1)
                    {
                        discreteCount++;
                    }
                }

                double discreteRatio = (double)discreteCount / halfLen;
                phaseDiffConsistency.Add(discreteRatio);
            }

            prevPhases = phases;
        }

        if (phaseDiffConsistency.Count == 0) return 0;

        double maxConsistency = 0;
        foreach (var c in phaseDiffConsistency)
        {
            if (c > maxConsistency) maxConsistency = c;
        }

        // High ratio of discrete phase differences = likely watermarked
        return Math.Min(1.0, Math.Max(0, (maxConsistency - 0.3) * 2));
    }

    private double DetectNaturalRecordingArtifacts(Vector<T> audio, int sampleRate)
    {
        // Natural recordings have: background noise, slight DC offset, frequency roll-off
        int frameSize = Math.Min(1024, audio.Length);
        var frame = new Vector<T>(frameSize);
        for (int i = 0; i < frameSize; i++)
        {
            frame[i] = audio[i];
        }

        // DC offset (natural recordings often have slight DC bias)
        double dcSum = 0;
        for (int i = 0; i < frameSize; i++)
        {
            dcSum += NumOps.ToDouble(frame[i]);
        }
        double dcOffset = Math.Abs(dcSum / frameSize);
        double hasDC = dcOffset > 0.001 ? 1.0 : 0;

        // Background noise floor (silent portions should have some noise)
        var sortedAbsValues = new double[frameSize];
        for (int i = 0; i < frameSize; i++)
        {
            sortedAbsValues[i] = Math.Abs(NumOps.ToDouble(frame[i]));
        }
        Array.Sort(sortedAbsValues);
        double noiseFloor = sortedAbsValues[frameSize / 10]; // 10th percentile
        double hasNoise = noiseFloor > 0.0001 ? 1.0 : 0;

        // Natural = has DC + has noise → score ~1.0
        return (hasDC + hasNoise) / 2.0;
    }
}
