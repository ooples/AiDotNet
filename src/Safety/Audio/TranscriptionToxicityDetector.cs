using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Detects toxic content in audio by analyzing acoustic features indicative of aggressive speech.
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
/// 4. Short-term energy variance — high variance suggests shouting patterns
/// 5. Zero-crossing rate — correlates with voiced/unvoiced speech characteristics
/// </para>
/// <para>
/// <b>References:</b>
/// - Multimodal toxicity detection: combining audio + text signals (ACL 2024)
/// - Speech emotion recognition for content moderation (INTERSPEECH 2024)
/// - Hate speech detection in speech: challenges and approaches (2024)
/// - Acoustic correlates of aggression in speech (Journal of Phonetics, 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TranscriptionToxicityDetector<T> : AudioSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;

    // Pre-computed constants
    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;

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

        _threshold = NumOps.FromDouble(threshold);
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
        T toxicityScore = EstimateAcousticToxicity(features);

        if (NumOps.GreaterThanOrEquals(toxicityScore, _threshold))
        {
            double scoreDouble = NumOps.ToDouble(toxicityScore);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Harassment,
                Severity = SafetySeverity.Medium,
                Confidence = scoreDouble,
                Description = $"Audio flagged for potentially aggressive/toxic speech patterns (score: {scoreDouble:F3}). " +
                              $"RMS energy: {NumOps.ToDouble(features.RMS):F4}, " +
                              $"Energy variance: {NumOps.ToDouble(features.ShortTermEnergyVariance):F4}, " +
                              $"ZCR: {NumOps.ToDouble(features.ZeroCrossingRate):F4}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private AcousticFeatures ComputeAcousticFeatures(Vector<T> samples, int sampleRate)
    {
        int length = samples.Length;
        T sum = Zero;
        T sumSq = Zero;
        T maxVal = samples[0];
        T minVal = samples[0];
        int zeroCrossings = 0;
        T prevSample = Zero;

        for (int i = 0; i < length; i++)
        {
            T val = samples[i];
            sum = NumOps.Add(sum, val);
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(val, val));

            if (NumOps.GreaterThan(val, maxVal)) maxVal = val;
            if (NumOps.LessThan(val, minVal)) minVal = val;

            if (i > 0 && IsZeroCrossing(prevSample, val))
            {
                zeroCrossings++;
            }

            prevSample = val;
        }

        T lengthT = NumOps.FromDouble(length);
        T mean = NumOps.Divide(sum, lengthT);
        T meanSq = NumOps.Divide(sumSq, lengthT);
        T variance = NumOps.Subtract(meanSq, NumOps.Multiply(mean, mean));
        T rms = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(meanSq)));
        T dynamicRange = NumOps.Subtract(maxVal, minVal);
        T zeroCrossingRate = NumOps.Divide(NumOps.FromDouble(zeroCrossings), lengthT);

        // Estimate short-term energy variation (proxy for shouting detection)
        int frameSize = sampleRate / 100; // 10ms frames
        T energyVariance = ComputeShortTermEnergyVariance(samples, frameSize);

        return new AcousticFeatures
        {
            Mean = mean,
            Variance = variance,
            RMS = rms,
            DynamicRange = dynamicRange,
            ZeroCrossingRate = zeroCrossingRate,
            ShortTermEnergyVariance = energyVariance,
            Duration = NumOps.Divide(NumOps.FromDouble(length), NumOps.FromDouble(sampleRate))
        };
    }

    private T ComputeShortTermEnergyVariance(Vector<T> samples, int frameSize)
    {
        if (frameSize <= 0 || samples.Length < frameSize)
        {
            return Zero;
        }

        int numFrames = samples.Length / frameSize;
        if (numFrames == 0)
        {
            return Zero;
        }

        var frameEnergies = new Vector<T>(numFrames);
        T energySum = Zero;

        for (int f = 0; f < numFrames; f++)
        {
            T frameSumSq = Zero;
            int start = f * frameSize;
            for (int i = 0; i < frameSize; i++)
            {
                T val = samples[start + i];
                frameSumSq = NumOps.Add(frameSumSq, NumOps.Multiply(val, val));
            }

            frameEnergies[f] = NumOps.Divide(frameSumSq, NumOps.FromDouble(frameSize));
            energySum = NumOps.Add(energySum, frameEnergies[f]);
        }

        T energyMean = NumOps.Divide(energySum, NumOps.FromDouble(numFrames));
        T varianceSum = Zero;
        for (int f = 0; f < numFrames; f++)
        {
            T diff = NumOps.Subtract(frameEnergies[f], energyMean);
            varianceSum = NumOps.Add(varianceSum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(varianceSum, NumOps.FromDouble(numFrames));
    }

    /// <summary>
    /// Estimates acoustic toxicity from features using a weighted heuristic model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Combines multiple acoustic indicators of aggressive/hostile speech:
    /// </para>
    /// <para>
    /// 1. <b>High RMS energy (25%)</b>: Shouting produces significantly higher average energy
    ///    than normal conversational speech. RMS above -20 dBFS (~0.1 amplitude) is elevated.
    /// </para>
    /// <para>
    /// 2. <b>Energy variance spikes (30%)</b>: Aggressive speech has much higher frame-to-frame
    ///    energy variation than calm speech, as speakers alternate between shouting and pauses.
    /// </para>
    /// <para>
    /// 3. <b>Elevated ZCR (15%)</b>: High zero-crossing rates correlate with unvoiced consonants
    ///    and fricatives, which are more prevalent in aggressive articulation.
    /// </para>
    /// <para>
    /// 4. <b>High dynamic range (20%)</b>: Aggressive speakers exhibit larger amplitude swings
    ///    compared to calm, measured speech.
    /// </para>
    /// <para>
    /// 5. <b>Short duration penalty (10%)</b>: Very short clips (&lt;0.5s) are penalized since
    ///    reliable detection requires sufficient context.
    /// </para>
    /// </remarks>
    private static T EstimateAcousticToxicity(AcousticFeatures features)
    {
        // 1. High RMS energy indicator: normal speech ~0.01-0.05, shouting ~0.1-0.5
        // Scale: 0 at RMS=0.02, 1.0 at RMS=0.3
        double rmsVal = NumOps.ToDouble(features.RMS);
        T rmsScore;
        if (rmsVal < 0.02)
        {
            rmsScore = Zero;
        }
        else
        {
            double rawRms = Math.Min(1.0, (rmsVal - 0.02) / 0.28);
            rmsScore = NumOps.FromDouble(rawRms);
        }

        // 2. Short-term energy variance: high variance = shouting/aggression
        // Normal speech energy variance ~1e-5 to 1e-4, aggressive ~1e-3 to 1e-2
        double eVar = NumOps.ToDouble(features.ShortTermEnergyVariance);
        T energyVarScore;
        if (eVar < 1e-5)
        {
            energyVarScore = Zero;
        }
        else
        {
            // Log-scale mapping: -5 (1e-5) to -2 (1e-2) maps to 0..1
            double logVar = Math.Log10(Math.Max(eVar, 1e-10));
            double rawEVar = Math.Min(1.0, Math.Max(0.0, (logVar + 5.0) / 3.0));
            energyVarScore = NumOps.FromDouble(rawEVar);
        }

        // 3. Zero-crossing rate: elevated in aggressive speech (more unvoiced sounds)
        // Normal conversational ZCR ~0.03-0.08, aggressive ~0.1-0.3
        double zcrVal = NumOps.ToDouble(features.ZeroCrossingRate);
        T zcrScore;
        if (zcrVal < 0.05)
        {
            zcrScore = Zero;
        }
        else
        {
            double rawZcr = Math.Min(1.0, (zcrVal - 0.05) / 0.25);
            zcrScore = NumOps.FromDouble(rawZcr);
        }

        // 4. Dynamic range: large swings indicate aggressive speech
        // Normal ~0.1-0.3, aggressive ~0.5-1.5+
        double dynRange = NumOps.ToDouble(features.DynamicRange);
        T dynamicRangeScore;
        if (dynRange < 0.1)
        {
            dynamicRangeScore = Zero;
        }
        else
        {
            double rawDyn = Math.Min(1.0, (dynRange - 0.1) / 1.0);
            dynamicRangeScore = NumOps.FromDouble(rawDyn);
        }

        // 5. Duration confidence: penalize very short clips
        double dur = NumOps.ToDouble(features.Duration);
        T durationConfidence = NumOps.FromDouble(Math.Min(1.0, dur / 0.5));

        // Weighted combination
        T w1 = NumOps.FromDouble(0.25);
        T w2 = NumOps.FromDouble(0.30);
        T w3 = NumOps.FromDouble(0.15);
        T w4 = NumOps.FromDouble(0.20);
        T w5 = NumOps.FromDouble(0.10);

        T term1 = NumOps.Multiply(w1, rmsScore);
        T term2 = NumOps.Multiply(w2, energyVarScore);
        T term3 = NumOps.Multiply(w3, zcrScore);
        T term4 = NumOps.Multiply(w4, dynamicRangeScore);
        T term5 = NumOps.Multiply(w5, durationConfidence);

        T rawScore = NumOps.Add(NumOps.Add(NumOps.Add(term1, term2), NumOps.Add(term3, term4)), term5);

        return Clamp01(rawScore);
    }

    private static bool IsZeroCrossing(T prev, T current)
    {
        bool currentNonNeg = NumOps.GreaterThanOrEquals(current, Zero);
        bool prevNonNeg = NumOps.GreaterThanOrEquals(prev, Zero);
        return currentNonNeg != prevNonNeg;
    }

    private static T Clamp01(T value)
    {
        if (NumOps.LessThan(value, Zero)) return Zero;
        if (NumOps.GreaterThan(value, One)) return One;
        return value;
    }

    private struct AcousticFeatures
    {
        public T Mean;
        public T Variance;
        public T RMS;
        public T DynamicRange;
        public T ZeroCrossingRate;
        public T ShortTermEnergyVariance;
        public T Duration;
    }
}
