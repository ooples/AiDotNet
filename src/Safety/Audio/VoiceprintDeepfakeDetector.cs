using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VoiceRadar: A Speech-Based Framework for Robust Voice Liveness Detection",
    "https://www.ndss-symposium.org/ndss-paper/voiceradar/",
    Year = 2025,
    Authors = "Various")]
public class VoiceprintDeepfakeDetector<T> : AudioSafetyModuleBase<T>
{

    private readonly T _threshold;
    private readonly int _frameSize;
    private readonly int _hopSize;

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
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < _frameSize * 4)
        {
            return findings; // Need enough frames for consistency analysis
        }

        using var noGrad = new NoGradScope<T>();

        // Extract voiceprint features for every frame at once.
        int numFrames = (audioSamples.Length - _frameSize) / _hopSize + 1;
        var frames = SafetyTensorOps<T>.Frames(SafetyTensorOps<T>.AsTensor(audioSamples), _frameSize, _hopSize, numFrames);
        var features = ExtractVoiceprintFeatures(frames, sampleRate);

        // Skip silent frames
        var voiced = SafetyTensorOps<T>.Gt(features.Energy, 1e-6);
        var kept = new VoiceprintFrames
        {
            Pitch = SafetyTensorOps<T>.Compact(features.Pitch, voiced),
            Energy = SafetyTensorOps<T>.Compact(features.Energy, voiced),
            SpectralCentroid = SafetyTensorOps<T>.Compact(features.SpectralCentroid, voiced),
            Formant1 = SafetyTensorOps<T>.Compact(features.Formant1, voiced),
            Formant2 = SafetyTensorOps<T>.Compact(features.Formant2, voiced)
        };

        if (kept.Energy.Length < 4) return findings;

        // 1. Pitch consistency: real voices have smooth pitch contour, clones may have jumps
        T pitchJitter = ComputePitchJitter(kept);

        // 2. Spectral envelope consistency: should be stable for same speaker
        T spectralConsistency = ComputeSpectralConsistency(kept);

        // 3. Formant stability: formants shift smoothly in real speech
        T formantStability = ComputeFormantStability(kept);

        // 4. Micro-variation analysis: real speech has natural micro-variations;
        //    cloned speech is often either too smooth or has artificial jitter
        T microVariation = ComputeMicroVariationAnomaly(kept);

        // Combined score
        T finalScoreT = NumOps.Add(
            NumOps.Add(NumOps.Multiply(NumOps.FromDouble(0.25), pitchJitter),
                NumOps.Multiply(NumOps.FromDouble(0.30), spectralConsistency)),
            NumOps.Add(NumOps.Multiply(NumOps.FromDouble(0.25), formantStability),
                NumOps.Multiply(NumOps.FromDouble(0.20), microVariation)));

        T thresholdVal = _threshold;
        if (NumOps.GreaterThanOrEquals(finalScoreT, thresholdVal))
        {
            double finalScore = NumOps.ToDouble(finalScoreT);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Voiceprint consistency analysis: potential voice deepfake " +
                              $"(score: {finalScore:F3}). Pitch jitter: {NumOps.ToDouble(pitchJitter):F3}, " +
                              $"spectral: {NumOps.ToDouble(spectralConsistency):F3}, formant: {NumOps.ToDouble(formantStability):F3}, " +
                              $"micro-variation: {NumOps.ToDouble(microVariation):F3}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private VoiceprintFrames ExtractVoiceprintFeatures(Tensor<T> frames, int sampleRate)
    {
        var engine = SafetyTensorOps<T>.Engine;
        var axis1 = new[] { 1 };

        // Hann-windowed frames; energy from the raw samples.
        var windowed = engine.TensorMultiply(frames, SafetyTensorOps<T>.SymmetricHann(_frameSize));
        var energy = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorSquare(frames), axis1, false), NumOps.FromDouble(_frameSize));

        // Magnitude spectrum, bins 0 .. halfLen-1
        int halfLen = _frameSize / 2;
        var magnitudes = engine.TensorNarrow(SafetyTensorOps<T>.FftMagnitude(windowed), 1, 0, halfLen);

        // Pitch estimation via autocorrelation
        var pitch = SafetyTensorOps<T>.AutocorrelationPitch(frames, sampleRate, Math.Min(sampleRate / 50, _frameSize / 2));

        // Spectral centroid over bins 1 .. halfLen-1
        var band = engine.TensorNarrow(magnitudes, 1, 1, halfLen - 1);
        var freqs = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(SafetyTensorOps<T>.Range(halfLen - 1), NumOps.One),
            NumOps.FromDouble((double)sampleRate / _frameSize));
        var weightedSum = engine.ReduceSum(engine.TensorMultiply(band, freqs), axis1, false);
        var magSum = engine.ReduceSum(band, axis1, false);
        var hasMagnitude = SafetyTensorOps<T>.Gt(magSum, 1e-10);
        var centroid = SafetyTensorOps<T>.Where(hasMagnitude,
            engine.TensorDivide(weightedSum, SafetyTensorOps<T>.Where(hasMagnitude, magSum, 1.0)), 0.0);

        // Simple formant estimation: find peaks in smoothed spectrum
        return new VoiceprintFrames
        {
            Pitch = pitch,
            Energy = energy,
            SpectralCentroid = centroid,
            Formant1 = FindFormantPeak(magnitudes, sampleRate, 200, 1000),
            Formant2 = FindFormantPeak(magnitudes, sampleRate, 800, 3000)
        };
    }

    private Tensor<T> FindFormantPeak(Tensor<T> magnitudes, int sampleRate, double minFreq, double maxFreq)
    {
        int frameCount = magnitudes.Shape[0];
        int minBin = (int)(minFreq * _frameSize / sampleRate);
        int maxBin = (int)(maxFreq * _frameSize / sampleRate);
        maxBin = Math.Min(maxBin, magnitudes.Shape[1] - 1);

        T binToHz = NumOps.FromDouble((double)sampleRate / _frameSize);
        if (minBin > maxBin)
        {
            // Empty search range: the peak stays at minBin.
            return SafetyTensorOps<T>.Full(new[] { frameCount }, NumOps.Multiply(NumOps.FromDouble(minBin), binToHz));
        }

        // First strictly larger magnitude above 0 wins; none -> minBin.
        var range = SafetyTensorOps<T>.Engine.TensorNarrow(magnitudes, 1, minBin, maxBin - minBin + 1);
        var (_, index) = SafetyTensorOps<T>.FirstStrictMax(range, 0.0);
        var peakBin = SafetyTensorOps<T>.Engine.TensorAddScalar(SafetyTensorOps<T>.MaxZero(index), NumOps.FromDouble(minBin));
        return SafetyTensorOps<T>.Engine.TensorMultiplyScalar(peakBin, binToHz);
    }

    private static T ComputePitchJitter(VoiceprintFrames frames)
    {
        var engine = SafetyTensorOps<T>.Engine;
        var pitches = SafetyTensorOps<T>.Compact(frames.Pitch, SafetyTensorOps<T>.Gt(frames.Pitch, 50.0));

        if (pitches.Length < 3) return NumOps.Zero;

        // Compute jitter: relative variation between consecutive pitch values
        T pitchMean = engine.TensorMean(pitches);
        T meanJitter = engine.TensorMean(engine.TensorAbs(engine.TensorDiff(pitches, 1, 0)));
        T relativeJitter = NumOps.GreaterThan(pitchMean, NumOps.FromDouble(1e-10))
            ? NumOps.Divide(meanJitter, pitchMean)
            : NumOps.Zero;

        // Natural speech: jitter ~0.01-0.03; cloned: often < 0.005 or > 0.05
        if (NumOps.LessThan(relativeJitter, NumOps.FromDouble(0.005)))
            return NumOps.Multiply(NumOps.Subtract(NumOps.FromDouble(0.005), relativeJitter), NumOps.FromDouble(200)); // Too stable
        if (NumOps.GreaterThan(relativeJitter, NumOps.FromDouble(0.05)))
            return SafetyTensorOps<T>.MinOne(NumOps.Multiply(NumOps.Subtract(relativeJitter, NumOps.FromDouble(0.05)), NumOps.FromDouble(20))); // Too jittery

        return NumOps.Zero;
    }

    private static T ComputeSpectralConsistency(VoiceprintFrames frames)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int count = frames.SpectralCentroid.Length;
        if (count < 3) return NumOps.Zero;

        // Compute coefficient of variation of spectral centroid
        T countT = NumOps.FromDouble(count);
        T mean = NumOps.Divide(engine.TensorSum(frames.SpectralCentroid), countT);
        T variance = NumOps.Subtract(
            NumOps.Divide(engine.TensorSumOfSquares(frames.SpectralCentroid), countT),
            NumOps.Multiply(mean, mean));
        T cv = NumOps.GreaterThan(mean, NumOps.FromDouble(1e-10))
            ? NumOps.Divide(NumOps.Sqrt(SafetyTensorOps<T>.MaxZero(variance)), mean)
            : NumOps.Zero;

        // Too consistent (robotic) or too variable (spliced)
        if (NumOps.LessThan(cv, NumOps.FromDouble(0.05)))
            return NumOps.Multiply(NumOps.Subtract(NumOps.FromDouble(0.05), cv), NumOps.FromDouble(20));
        if (NumOps.GreaterThan(cv, NumOps.FromDouble(0.4)))
            return SafetyTensorOps<T>.MinOne(NumOps.Multiply(NumOps.Subtract(cv, NumOps.FromDouble(0.4)), NumOps.FromDouble(2)));

        return NumOps.Zero;
    }

    private static T ComputeFormantStability(VoiceprintFrames frames)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int count = frames.Formant1.Length;
        if (count < 3) return NumOps.Zero;

        // Check for sudden formant jumps (splicing artifacts):
        // jumps > 200 Hz (F1) or > 400 Hz (F2) between consecutive frames are suspicious
        var f1Jump = SafetyTensorOps<T>.Gt(engine.TensorAbs(engine.TensorDiff(frames.Formant1, 1, 0)), 200.0);
        var f2Jump = SafetyTensorOps<T>.Gt(engine.TensorAbs(engine.TensorDiff(frames.Formant2, 1, 0)), 400.0);
        T jumpCount = engine.TensorSum(SafetyTensorOps<T>.Or(f1Jump, f2Jump));

        T jumpRate = NumOps.Divide(jumpCount, NumOps.FromDouble(count - 1));

        // Natural: jumpRate < 0.1; cloned/spliced: often > 0.2
        return SafetyTensorOps<T>.ClampUnit(NumOps.Multiply(NumOps.Subtract(jumpRate, NumOps.FromDouble(0.1)), NumOps.FromDouble(5)));
    }

    private static T ComputeMicroVariationAnomaly(VoiceprintFrames frames)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int count = frames.Energy.Length;
        if (count < 5) return NumOps.Zero;

        // Compute frame-to-frame energy variation
        var current = engine.TensorNarrow(frames.Energy, 0, 1, count - 1);
        var previous = engine.TensorNarrow(frames.Energy, 0, 0, count - 1);
        var diff = engine.TensorAbs(engine.TensorSubtract(current, previous));
        var avg = engine.TensorDivideScalar(engine.TensorAdd(current, previous), NumOps.FromDouble(2));
        var hasEnergy = SafetyTensorOps<T>.Gt(avg, 1e-10);
        var diffs = SafetyTensorOps<T>.Where(hasEnergy, engine.TensorDivide(diff, SafetyTensorOps<T>.Where(hasEnergy, avg, 1.0)), 0.0);

        var (meanDiff, varDiff) = SafetyTensorOps<T>.MeanVariance(diffs);

        T cvDiff = NumOps.GreaterThan(meanDiff, NumOps.FromDouble(1e-10))
            ? NumOps.Divide(NumOps.Sqrt(varDiff), meanDiff)
            : NumOps.Zero;

        // Too uniform energy changes (synthetic) or too erratic
        if (NumOps.LessThan(cvDiff, NumOps.FromDouble(0.3)))
            return NumOps.Multiply(NumOps.Subtract(NumOps.FromDouble(0.3), cvDiff), NumOps.FromDouble(3));
        if (NumOps.GreaterThan(cvDiff, NumOps.FromDouble(2.0)))
            return SafetyTensorOps<T>.MinOne(NumOps.Divide(NumOps.Subtract(cvDiff, NumOps.FromDouble(2.0)), NumOps.FromDouble(3)));

        return NumOps.Zero;
    }

    /// <summary>Per-frame voiceprint features, one element per frame.</summary>
    private sealed class VoiceprintFrames
    {
        public Tensor<T> Pitch { get; set; } = new Tensor<T>(new[] { 0 });
        public Tensor<T> Energy { get; set; } = new Tensor<T>(new[] { 0 });
        public Tensor<T> SpectralCentroid { get; set; } = new Tensor<T>(new[] { 0 });
        public Tensor<T> Formant1 { get; set; } = new Tensor<T>(new[] { 0 });
        public Tensor<T> Formant2 { get; set; } = new Tensor<T>(new[] { 0 });
    }
}
