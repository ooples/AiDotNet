using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Detects deepfake audio by analyzing speaker voiceprint consistency throughout the recording.
/// </summary>
/// <remarks>
/// <para>
/// Extracts short-term speaker embeddings from overlapping frames of the audio and measures
/// consistency across the recording. Real speech has stable speaker characteristics (pitch
/// contour, formant patterns, spectral envelope) while cloned/synthesized voices often show
/// temporal inconsistencies in these features — either too stable (robotic) or with sudden
/// jumps (spliced segments).
/// </para>
/// <para>
/// <b>For Beginners:</b> Each person's voice has unique characteristics — like a fingerprint.
/// Real speech keeps these characteristics consistent. Voice cloning often produces subtle
/// inconsistencies that this module detects by comparing "voice fingerprints" from different
/// parts of the recording.
/// </para>
/// <para>
/// <b>References:</b>
/// - VoiceRadar: Voice deepfake detection framework (NDSS 2025)
/// - SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
/// - Voice cloning detection via speaker verification mismatch (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VoiceprintDeepfakeDetector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;
    private readonly int _frameSize;
    private readonly int _hopSize;
    private readonly FastFourierTransform<T> _fft;

    /// <inheritdoc />
    public override string ModuleName => "VoiceprintDeepfakeDetector";

    /// <summary>
    /// Initializes a new voiceprint deepfake detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public VoiceprintDeepfakeDetector(double threshold = 0.5, int sampleRate = 16000)
        : base(sampleRate)
    {
        _threshold = NumOps.FromDouble(threshold);
        _frameSize = 512;
        _hopSize = 256;
        _fft = new FastFourierTransform<T>();
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < _frameSize * 4)
        {
            return findings; // Need enough frames for consistency analysis
        }

        // Extract voiceprint features per frame
        var frameFeatures = new List<VoiceprintFrame>();
        int numFrames = (audioSamples.Length - _frameSize) / _hopSize + 1;

        for (int i = 0; i < numFrames; i++)
        {
            int start = i * _hopSize;
            if (start + _frameSize > audioSamples.Length) break;

            var frame = ExtractVoiceprintFeatures(audioSamples, start, sampleRate);
            if (frame.Energy > 1e-6) // Skip silent frames
            {
                frameFeatures.Add(frame);
            }
        }

        if (frameFeatures.Count < 4) return findings;

        // 1. Pitch consistency: real voices have smooth pitch contour, clones may have jumps
        double pitchJitter = ComputePitchJitter(frameFeatures);

        // 2. Spectral envelope consistency: should be stable for same speaker
        double spectralConsistency = ComputeSpectralConsistency(frameFeatures);

        // 3. Formant stability: formants shift smoothly in real speech
        double formantStability = ComputeFormantStability(frameFeatures);

        // 4. Micro-variation analysis: real speech has natural micro-variations;
        //    cloned speech is often either too smooth or has artificial jitter
        double microVariation = ComputeMicroVariationAnomaly(frameFeatures);

        // Combined score
        double finalScore = 0.25 * pitchJitter +
                           0.30 * spectralConsistency +
                           0.25 * formantStability +
                           0.20 * microVariation;

        T thresholdVal = _threshold;
        if (NumOps.GreaterThanOrEquals(NumOps.FromDouble(finalScore), thresholdVal))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Voiceprint consistency analysis: potential voice deepfake " +
                              $"(score: {finalScore:F3}). Pitch jitter: {pitchJitter:F3}, " +
                              $"spectral: {spectralConsistency:F3}, formant: {formantStability:F3}, " +
                              $"micro-variation: {microVariation:F3}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private VoiceprintFrame ExtractVoiceprintFeatures(Vector<T> audio, int start, int sampleRate)
    {
        var frame = new VoiceprintFrame();

        // Extract windowed frame
        var windowed = new Vector<T>(_frameSize);
        T energySum = NumOps.Zero;
        for (int i = 0; i < _frameSize; i++)
        {
            // Hann window
            double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (_frameSize - 1)));
            T sample = audio[start + i];
            windowed[i] = NumOps.Multiply(sample, NumOps.FromDouble(w));
            energySum = NumOps.Add(energySum, NumOps.Multiply(sample, sample));
        }
        frame.Energy = NumOps.ToDouble(energySum) / _frameSize;

        // FFT for spectral features
        var spectrum = _fft.Forward(windowed);
        int halfLen = spectrum.Length / 2;

        // Magnitude spectrum
        var magnitudes = new double[halfLen];
        for (int k = 0; k < halfLen; k++)
        {
            magnitudes[k] = NumOps.ToDouble(spectrum[k].Magnitude);
        }

        // Pitch estimation via autocorrelation
        frame.Pitch = EstimatePitch(audio, start, sampleRate);

        // Spectral centroid
        double weightedSum = 0, magSum = 0;
        for (int k = 1; k < halfLen; k++)
        {
            double freq = (double)k * sampleRate / _frameSize;
            weightedSum += freq * magnitudes[k];
            magSum += magnitudes[k];
        }
        frame.SpectralCentroid = magSum > 1e-10 ? weightedSum / magSum : 0;

        // Spectral rolloff (85%)
        double cumSum = 0;
        double targetSum = magSum * 0.85;
        frame.SpectralRolloff = 0;
        for (int k = 1; k < halfLen; k++)
        {
            cumSum += magnitudes[k];
            if (cumSum >= targetSum)
            {
                frame.SpectralRolloff = (double)k * sampleRate / _frameSize;
                break;
            }
        }

        // Simple formant estimation: find peaks in smoothed spectrum
        frame.Formant1 = FindFormantPeak(magnitudes, sampleRate, 200, 1000);
        frame.Formant2 = FindFormantPeak(magnitudes, sampleRate, 800, 3000);

        return frame;
    }

    private double EstimatePitch(Vector<T> audio, int start, int sampleRate)
    {
        // Autocorrelation-based pitch detection
        int minLag = sampleRate / 500; // Max pitch 500 Hz
        int maxLag = sampleRate / 50;  // Min pitch 50 Hz
        maxLag = Math.Min(maxLag, _frameSize / 2);

        if (minLag >= maxLag) return 0;

        double maxCorr = 0;
        int bestLag = 0;

        for (int lag = minLag; lag < maxLag; lag++)
        {
            double corr = 0;
            double norm1 = 0, norm2 = 0;
            int len = _frameSize - lag;

            for (int i = 0; i < len; i++)
            {
                double v1 = NumOps.ToDouble(audio[start + i]);
                double v2 = NumOps.ToDouble(audio[start + i + lag]);
                corr += v1 * v2;
                norm1 += v1 * v1;
                norm2 += v2 * v2;
            }

            double denom = Math.Sqrt(norm1 * norm2);
            double normalized = denom > 1e-10 ? corr / denom : 0;

            if (normalized > maxCorr)
            {
                maxCorr = normalized;
                bestLag = lag;
            }
        }

        return bestLag > 0 && maxCorr > 0.3 ? (double)sampleRate / bestLag : 0;
    }

    private double FindFormantPeak(double[] magnitudes, int sampleRate, double minFreq, double maxFreq)
    {
        int minBin = (int)(minFreq * _frameSize / sampleRate);
        int maxBin = (int)(maxFreq * _frameSize / sampleRate);
        maxBin = Math.Min(maxBin, magnitudes.Length - 1);

        double maxMag = 0;
        int peakBin = minBin;

        for (int k = minBin; k <= maxBin; k++)
        {
            if (magnitudes[k] > maxMag)
            {
                maxMag = magnitudes[k];
                peakBin = k;
            }
        }

        return (double)peakBin * sampleRate / _frameSize;
    }

    private static double ComputePitchJitter(List<VoiceprintFrame> frames)
    {
        var pitches = new List<double>();
        foreach (var f in frames)
        {
            if (f.Pitch > 50) pitches.Add(f.Pitch);
        }

        if (pitches.Count < 3) return 0;

        // Compute jitter: relative variation between consecutive pitch values
        double jitterSum = 0;
        double pitchMean = 0;
        foreach (var p in pitches) pitchMean += p;
        pitchMean /= pitches.Count;

        for (int i = 1; i < pitches.Count; i++)
        {
            jitterSum += Math.Abs(pitches[i] - pitches[i - 1]);
        }
        double meanJitter = jitterSum / (pitches.Count - 1);
        double relativeJitter = pitchMean > 1e-10 ? meanJitter / pitchMean : 0;

        // Natural speech: jitter ~0.01-0.03; cloned: often < 0.005 or > 0.05
        if (relativeJitter < 0.005)
            return (0.005 - relativeJitter) * 200; // Too stable
        if (relativeJitter > 0.05)
            return Math.Min(1.0, (relativeJitter - 0.05) * 20); // Too jittery

        return 0;
    }

    private static double ComputeSpectralConsistency(List<VoiceprintFrame> frames)
    {
        if (frames.Count < 3) return 0;

        // Compute coefficient of variation of spectral centroid
        double sum = 0, sumSq = 0;
        foreach (var f in frames)
        {
            sum += f.SpectralCentroid;
            sumSq += f.SpectralCentroid * f.SpectralCentroid;
        }

        double mean = sum / frames.Count;
        double variance = sumSq / frames.Count - mean * mean;
        double cv = mean > 1e-10 ? Math.Sqrt(Math.Max(0, variance)) / mean : 0;

        // Too consistent (robotic) or too variable (spliced)
        if (cv < 0.05) return (0.05 - cv) * 20;
        if (cv > 0.4) return Math.Min(1.0, (cv - 0.4) * 2);

        return 0;
    }

    private static double ComputeFormantStability(List<VoiceprintFrame> frames)
    {
        if (frames.Count < 3) return 0;

        // Check for sudden formant jumps (splicing artifacts)
        int jumpCount = 0;
        for (int i = 1; i < frames.Count; i++)
        {
            double f1Diff = Math.Abs(frames[i].Formant1 - frames[i - 1].Formant1);
            double f2Diff = Math.Abs(frames[i].Formant2 - frames[i - 1].Formant2);

            // Formant jumps > 200 Hz between consecutive frames are suspicious
            if (f1Diff > 200 || f2Diff > 400)
            {
                jumpCount++;
            }
        }

        double jumpRate = (double)jumpCount / (frames.Count - 1);

        // Natural: jumpRate < 0.1; cloned/spliced: often > 0.2
        return Math.Min(1.0, Math.Max(0, (jumpRate - 0.1) * 5));
    }

    private static double ComputeMicroVariationAnomaly(List<VoiceprintFrame> frames)
    {
        if (frames.Count < 5) return 0;

        // Compute frame-to-frame energy variation
        var diffs = new List<double>();
        for (int i = 1; i < frames.Count; i++)
        {
            double diff = Math.Abs(frames[i].Energy - frames[i - 1].Energy);
            double avg = (frames[i].Energy + frames[i - 1].Energy) / 2;
            diffs.Add(avg > 1e-10 ? diff / avg : 0);
        }

        double meanDiff = 0;
        foreach (var d in diffs) meanDiff += d;
        meanDiff /= diffs.Count;

        double varDiff = 0;
        foreach (var d in diffs)
        {
            double dev = d - meanDiff;
            varDiff += dev * dev;
        }
        varDiff /= diffs.Count;

        double cvDiff = meanDiff > 1e-10 ? Math.Sqrt(varDiff) / meanDiff : 0;

        // Too uniform energy changes (synthetic) or too erratic
        if (cvDiff < 0.3) return (0.3 - cvDiff) * 3;
        if (cvDiff > 2.0) return Math.Min(1.0, (cvDiff - 2.0) / 3);

        return 0;
    }

    private struct VoiceprintFrame
    {
        public double Pitch;
        public double Energy;
        public double SpectralCentroid;
        public double SpectralRolloff;
        public double Formant1;
        public double Formant2;
    }
}
