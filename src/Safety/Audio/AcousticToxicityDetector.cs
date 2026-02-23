using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Detects toxic/aggressive speech patterns directly from acoustic features without transcription.
/// </summary>
/// <remarks>
/// <para>
/// Analyzes prosodic and spectral features that correlate with aggression, shouting, and hostile
/// intent. Aggressive speech has characteristic acoustic signatures: elevated pitch, high energy
/// variance, rapid pitch changes, and spectral centroid shifts. This approach works regardless
/// of language and catches toxicity even when words are unintelligible.
/// </para>
/// <para>
/// <b>For Beginners:</b> You can often tell someone is angry or threatening just from the
/// tone of their voice — even if you don't understand the language. This module does the same:
/// it analyzes voice patterns like loudness changes, pitch, and speaking rate to detect
/// aggressive or hostile speech.
/// </para>
/// <para>
/// <b>References:</b>
/// - Taxonomy of speech generator harms including swatting attacks (2024, arxiv:2402.01708)
/// - Acoustic emotion recognition using prosodic and spectral features (2023)
/// - Paralinguistic analysis for aggression detection (INTERSPEECH 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AcousticToxicityDetector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;
    private readonly FastFourierTransform<T> _fft;

    /// <inheritdoc />
    public override string ModuleName => "AcousticToxicityDetector";

    /// <summary>
    /// Initializes a new acoustic toxicity detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public AcousticToxicityDetector(double threshold = 0.5, int sampleRate = 16000)
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

        int frameSize = 512;
        int hopSize = 256;
        int numFrames = (audioSamples.Length - frameSize) / hopSize + 1;

        // Extract per-frame features
        var energies = new List<double>();
        var pitches = new List<double>();
        var centroids = new List<double>();
        var zcrs = new List<double>();

        for (int f = 0; f < numFrames; f++)
        {
            int start = f * hopSize;
            if (start + frameSize > audioSamples.Length) break;

            // Frame energy
            double energy = 0;
            double zcr = 0;
            for (int i = 0; i < frameSize; i++)
            {
                double val = NumOps.ToDouble(audioSamples[start + i]);
                energy += val * val;
                if (i > 0)
                {
                    double prev = NumOps.ToDouble(audioSamples[start + i - 1]);
                    if ((val >= 0 && prev < 0) || (val < 0 && prev >= 0))
                        zcr += 1.0;
                }
            }
            energy /= frameSize;
            zcr /= frameSize;

            if (energy < 1e-8) continue; // Skip silent frames

            energies.Add(energy);
            zcrs.Add(zcr);

            // Spectral centroid via FFT
            var windowed = new Vector<T>(frameSize);
            for (int i = 0; i < frameSize; i++)
            {
                double w = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (frameSize - 1)));
                windowed[i] = NumOps.Multiply(audioSamples[start + i], NumOps.FromDouble(w));
            }

            var spectrum = _fft.Forward(windowed);
            int halfLen = spectrum.Length / 2;

            double weightedSum = 0, magSum = 0;
            for (int k = 1; k < halfLen; k++)
            {
                double mag = NumOps.ToDouble(spectrum[k].Magnitude);
                weightedSum += k * mag;
                magSum += mag;
            }
            centroids.Add(magSum > 1e-10 ? weightedSum / magSum * sampleRate / frameSize : 0);

            // Pitch estimation
            double pitch = EstimatePitch(audioSamples, start, frameSize, sampleRate);
            if (pitch > 0) pitches.Add(pitch);
        }

        if (energies.Count < 3) return findings;

        // Feature 1: Energy dynamics (30%) — shouting/aggression has high and variable energy
        double energyScore = ComputeEnergyAggressionScore(energies);

        // Feature 2: Pitch characteristics (25%) — elevated, variable pitch indicates anger
        double pitchScore = ComputePitchAggressionScore(pitches);

        // Feature 3: Spectral characteristics (25%) — high centroid = harsh/aggressive timbre
        double spectralScore = ComputeSpectralAggressionScore(centroids);

        // Feature 4: Speaking rate (20%) — high ZCR correlates with fast/aggressive speech
        double rateScore = ComputeSpeechRateScore(zcrs);

        double finalScore = 0.30 * energyScore +
                           0.25 * pitchScore +
                           0.25 * spectralScore +
                           0.20 * rateScore;

        if (NumOps.GreaterThanOrEquals(NumOps.FromDouble(finalScore), _threshold))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.ViolenceThreat,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Acoustic toxicity detected (score: {finalScore:F3}). " +
                              $"Energy aggression: {energyScore:F3}, pitch: {pitchScore:F3}, " +
                              $"spectral: {spectralScore:F3}, speech rate: {rateScore:F3}. " +
                              $"Prosodic features indicate aggressive/hostile speech.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static double ComputeEnergyAggressionScore(List<double> energies)
    {
        double mean = 0, max = 0;
        foreach (var e in energies)
        {
            mean += e;
            if (e > max) max = e;
        }
        mean /= energies.Count;

        double variance = 0;
        foreach (var e in energies)
        {
            double diff = e - mean;
            variance += diff * diff;
        }
        variance /= energies.Count;

        // High energy + high variance = aggressive
        // RMS energy for speech typically ~0.01-0.05; shouting ~0.1-0.5
        double rms = Math.Sqrt(mean);
        double energyIndicator = Math.Min(1.0, rms / 0.3);

        // Energy variance (dynamic range)
        double cv = mean > 1e-10 ? Math.Sqrt(variance) / mean : 0;
        double dynamicIndicator = Math.Min(1.0, cv / 2);

        return 0.6 * energyIndicator + 0.4 * dynamicIndicator;
    }

    private static double ComputePitchAggressionScore(List<double> pitches)
    {
        if (pitches.Count < 2) return 0;

        double mean = 0;
        foreach (var p in pitches) mean += p;
        mean /= pitches.Count;

        // High pitch mean (>200 Hz) indicates agitation
        // Normal conversational: 100-150 Hz (male), 180-250 Hz (female)
        // Aggressive: often > 200 Hz (male), > 300 Hz (female)
        double pitchElevation = Math.Min(1.0, Math.Max(0, (mean - 180) / 200));

        // Pitch variance (rapid changes indicate anger)
        double variance = 0;
        foreach (var p in pitches)
        {
            double diff = p - mean;
            variance += diff * diff;
        }
        variance /= pitches.Count;
        double pitchVariability = Math.Min(1.0, Math.Sqrt(variance) / 100);

        return 0.5 * pitchElevation + 0.5 * pitchVariability;
    }

    private static double ComputeSpectralAggressionScore(List<double> centroids)
    {
        if (centroids.Count == 0) return 0;

        double mean = 0;
        foreach (var c in centroids) mean += c;
        mean /= centroids.Count;

        // High spectral centroid = harsh, aggressive timbre
        // Normal speech: ~1000-2000 Hz; shouting: ~2000-4000 Hz
        double centroidScore = Math.Min(1.0, Math.Max(0, (mean - 1500) / 2500));

        return centroidScore;
    }

    private static double ComputeSpeechRateScore(List<double> zcrs)
    {
        if (zcrs.Count == 0) return 0;

        double mean = 0;
        foreach (var z in zcrs) mean += z;
        mean /= zcrs.Count;

        // High ZCR indicates fast articulation
        // Normal: ~0.05-0.1; fast/aggressive: ~0.15-0.3
        return Math.Min(1.0, Math.Max(0, (mean - 0.1) / 0.2));
    }

    private static double EstimatePitch(Vector<T> audio, int start, int frameSize, int sampleRate)
    {
        int minLag = sampleRate / 500;
        int maxLag = Math.Min(sampleRate / 50, frameSize / 2);
        if (minLag >= maxLag) return 0;

        double maxCorr = 0;
        int bestLag = 0;

        for (int lag = minLag; lag < maxLag; lag++)
        {
            double corr = 0, norm1 = 0, norm2 = 0;
            int len = frameSize - lag;
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
            if (normalized > maxCorr) { maxCorr = normalized; bestLag = lag; }
        }

        return bestLag > 0 && maxCorr > 0.3 ? (double)sampleRate / bestLag : 0;
    }
}
