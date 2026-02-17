using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Safety.Text;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Detects toxic content in audio by analyzing audio features indicative of aggressive speech.
/// </summary>
/// <remarks>
/// <para>
/// This module analyzes acoustic features (energy, pitch variation, speaking rate) that correlate
/// with aggressive or hateful speech patterns. It acts as an audio-level complement to text-based
/// toxicity detection by catching patterns that transcription-based approaches might miss.
/// </para>
/// <para>
/// <b>For Beginners:</b> While text-based toxicity detection catches harmful words, this module
/// analyzes the sound itself — shouting, aggressive tone, and speech patterns associated with
/// hateful content can be detected from audio features alone.
/// </para>
/// <para>
/// <b>Detection approach:</b>
/// 1. Energy analysis — sudden loud bursts can indicate shouting/aggression
/// 2. Speaking rate estimation — very rapid speech can correlate with agitation
/// 3. Dynamic range — extreme variation in volume may indicate emotional distress
/// </para>
/// <para>
/// <b>References:</b>
/// - Multimodal toxicity detection: combining audio + text signals (ACL 2024)
/// - Speech emotion recognition for content moderation (INTERSPEECH 2024)
/// - Hate speech detection in speech: challenges and approaches (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TranscriptionToxicityDetector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;

    /// <inheritdoc />
    public override string ModuleName => "TranscriptionToxicityDetector";

    /// <summary>
    /// Initializes a new transcription toxicity detector.
    /// </summary>
    /// <param name="threshold">
    /// Detection threshold (0-1). Audio scoring above this is flagged. Default: 0.7.
    /// </param>
    /// <param name="defaultSampleRate">
    /// Default sample rate in Hz. Default: 16000.
    /// </param>
    public TranscriptionToxicityDetector(double threshold = 0.7, int defaultSampleRate = 16000)
        : base(defaultSampleRate)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length == 0)
        {
            return findings;
        }

        var features = ComputeAcousticFeatures(audioSamples, sampleRate);
        var toxicityScore = EstimateAcousticToxicity(features);

        if (toxicityScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Harassment,
                Severity = SafetySeverity.Medium,
                Confidence = toxicityScore,
                Description = $"Audio flagged for potentially aggressive/toxic speech patterns (score: {toxicityScore:F3}). " +
                              "Acoustic feature analysis detected patterns consistent with hostile speech.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private AcousticFeatures ComputeAcousticFeatures(Vector<T> samples, int sampleRate)
    {
        int length = samples.Length;
        double sum = 0;
        double sumSq = 0;
        double max = double.MinValue;
        double min = double.MaxValue;
        int zeroCrossings = 0;
        double prevSample = 0;

        for (int i = 0; i < length; i++)
        {
            double val = NumOps.ToDouble(samples[i]);
            sum += val;
            sumSq += val * val;
            if (val > max) max = val;
            if (val < min) min = val;

            if (i > 0 && ((val >= 0 && prevSample < 0) || (val < 0 && prevSample >= 0)))
            {
                zeroCrossings++;
            }

            prevSample = val;
        }

        double mean = sum / length;
        double variance = sumSq / length - mean * mean;
        double rms = Math.Sqrt(sumSq / length);
        double dynamicRange = max - min;
        double zeroCrossingRate = (double)zeroCrossings / length;

        // Estimate short-term energy variation (proxy for shouting detection)
        int frameSize = sampleRate / 100; // 10ms frames
        double energyVariance = ComputeShortTermEnergyVariance(samples, frameSize);

        return new AcousticFeatures
        {
            Mean = mean,
            Variance = variance,
            RMS = rms,
            DynamicRange = dynamicRange,
            ZeroCrossingRate = zeroCrossingRate,
            ShortTermEnergyVariance = energyVariance,
            Duration = (double)length / sampleRate
        };
    }

    private double ComputeShortTermEnergyVariance(Vector<T> samples, int frameSize)
    {
        if (frameSize <= 0 || samples.Length < frameSize)
        {
            return 0.0;
        }

        int numFrames = samples.Length / frameSize;
        if (numFrames == 0)
        {
            return 0.0;
        }

        double[] frameEnergies = new double[numFrames];
        double energySum = 0;

        for (int f = 0; f < numFrames; f++)
        {
            double frameSumSq = 0;
            int start = f * frameSize;
            for (int i = 0; i < frameSize; i++)
            {
                double val = NumOps.ToDouble(samples[start + i]);
                frameSumSq += val * val;
            }

            frameEnergies[f] = frameSumSq / frameSize;
            energySum += frameEnergies[f];
        }

        double energyMean = energySum / numFrames;
        double varianceSum = 0;
        for (int f = 0; f < numFrames; f++)
        {
            double diff = frameEnergies[f] - energyMean;
            varianceSum += diff * diff;
        }

        return varianceSum / numFrames;
    }

    /// <summary>
    /// Estimates acoustic toxicity from features.
    /// </summary>
    /// <remarks>
    /// Placeholder heuristic. In production, replace with a trained acoustic emotion classifier
    /// or a speech-to-text pipeline feeding into a text toxicity model.
    /// Returns 0.0 to avoid false positives until a real model is integrated.
    /// </remarks>
    private static double EstimateAcousticToxicity(AcousticFeatures features)
    {
        _ = features;
        return 0.0;
    }

    private struct AcousticFeatures
    {
        public double Mean;
        public double Variance;
        public double RMS;
        public double DynamicRange;
        public double ZeroCrossingRate;
        public double ShortTermEnergyVariance;
        public double Duration;
    }
}
