using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Not My Voice! A Taxonomy of Ethical and Safety Harms of Speech Generators",
    "https://arxiv.org/abs/2402.01708",
    Year = 2024,
    Authors = "Various")]
public class AcousticToxicityDetector<T> : AudioSafetyModuleBase<T>
{

    private readonly T _threshold;

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
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 1024)
        {
            return findings;
        }

        using var noGrad = new NoGradScope<T>();
        var engine = SafetyTensorOps<T>.Engine;
        int frameSize = 512;
        int hopSize = 256;
        int numFrames = (audioSamples.Length - frameSize) / hopSize + 1;

        // Per-frame features are extracted in batches of FrameBatch frames, so the frame, spectrum and lag tensors
        // of a long recording are never held at once; only the per-frame values are kept.
        var audio = SafetyTensorOps<T>.AsTensor(audioSamples);
        var energyParts = new List<Tensor<T>>();
        var zcrParts = new List<Tensor<T>>();
        var centroidParts = new List<Tensor<T>>();
        var pitchParts = new List<Tensor<T>>();
        for (int first = 0; first < numFrames; first += FrameBatch)
        {
            int count = Math.Min(FrameBatch, numFrames - first);
            var segment = engine.TensorNarrow(audio, 0, first * hopSize, (count - 1) * hopSize + frameSize);
            var (energy, zcr, centroid, pitch, voiced) = FrameFeatures(
                SafetyTensorOps<T>.Frames(segment, frameSize, hopSize, count), sampleRate);
            energyParts.Add(SafetyTensorOps<T>.Compact(energy, voiced));
            zcrParts.Add(SafetyTensorOps<T>.Compact(zcr, voiced));
            centroidParts.Add(SafetyTensorOps<T>.Compact(centroid, voiced));
            pitchParts.Add(SafetyTensorOps<T>.Compact(pitch, SafetyTensorOps<T>.And(voiced, SafetyTensorOps<T>.Gt(pitch, 0.0))));
        }

        var energies = Join(energyParts);
        if (energies.Length < 3) return findings;
        var zcrs = Join(zcrParts);
        var centroids = Join(centroidParts);
        var pitches = Join(pitchParts);

        // Feature 1: Energy dynamics (30%) — shouting/aggression has high and variable energy
        T energyScore = ComputeEnergyAggressionScore(energies);

        // Feature 2: Pitch characteristics (25%) — elevated, variable pitch indicates anger
        T pitchScore = ComputePitchAggressionScore(pitches);

        // Feature 3: Spectral characteristics (25%) — high centroid = harsh/aggressive timbre
        T spectralScore = ComputeSpectralAggressionScore(centroids);

        // Feature 4: Speaking rate (20%) — high ZCR correlates with fast/aggressive speech
        T rateScore = ComputeSpeechRateScore(zcrs);

        T finalScoreT = NumOps.Add(
            NumOps.Add(NumOps.Multiply(NumOps.FromDouble(0.30), energyScore),
                NumOps.Multiply(NumOps.FromDouble(0.25), pitchScore)),
            NumOps.Add(NumOps.Multiply(NumOps.FromDouble(0.25), spectralScore),
                NumOps.Multiply(NumOps.FromDouble(0.20), rateScore)));

        if (NumOps.GreaterThanOrEquals(finalScoreT, _threshold))
        {
            double finalScore = NumOps.ToDouble(finalScoreT);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.ViolenceThreat,
                Severity = finalScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Acoustic toxicity detected (score: {finalScore:F3}). " +
                              $"Energy aggression: {NumOps.ToDouble(energyScore):F3}, pitch: {NumOps.ToDouble(pitchScore):F3}, " +
                              $"spectral: {NumOps.ToDouble(spectralScore):F3}, speech rate: {NumOps.ToDouble(rateScore):F3}. " +
                              $"Prosodic features indicate aggressive/hostile speech.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>The number of frames analysed per tensor batch.</summary>
    internal const int FrameBatch = 1024;

    /// <summary>
    /// Energy (mean square), zero-crossing rate, spectral centroid and autocorrelation pitch of every frame of a
    /// <c>[frames, frameSize]</c> batch, with the voiced mask (energy not below 1e-8; a NaN energy is kept).
    /// </summary>
    private static (Tensor<T> Energy, Tensor<T> Zcr, Tensor<T> Centroid, Tensor<T> Pitch, Tensor<T> Voiced) FrameFeatures(
        Tensor<T> frames, int sampleRate)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int frameSize = frames.Shape[1];
        var axis1 = new[] { 1 };

        // Frame energy (mean square) and zero-crossing rate.
        var energy = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorSquare(frames), axis1, false), NumOps.FromDouble(frameSize));
        var zcr = engine.TensorDivideScalar(
            SafetyTensorOps<T>.ZeroCrossings(frames, nanIsNegative: false), NumOps.FromDouble(frameSize));

        // Skip silent frames (energy < 1e-8; a NaN energy is not "< 1e-8" and is kept).
        var voiced = SafetyTensorOps<T>.Not(SafetyTensorOps<T>.Lt(energy, 1e-8));

        // Spectral centroid via FFT of the Hann-windowed frame, bins 1 .. halfLen-1.
        var windowed = engine.TensorMultiply(frames, SafetyTensorOps<T>.SymmetricHann(frameSize));
        var magnitudes = SafetyTensorOps<T>.FftMagnitude(windowed);
        int halfLen = frameSize / 2;
        var band = engine.TensorNarrow(magnitudes, 1, 1, halfLen - 1);
        var binIndex = engine.TensorAddScalar(SafetyTensorOps<T>.Range(halfLen - 1), NumOps.One);
        var weightedSum = engine.ReduceSum(engine.TensorMultiply(band, binIndex), axis1, false);
        var magSum = engine.ReduceSum(band, axis1, false);
        var hasMagnitude = SafetyTensorOps<T>.Gt(magSum, 1e-10);
        var centroid = SafetyTensorOps<T>.Where(hasMagnitude,
            engine.TensorMultiplyScalar(
                engine.TensorDivide(weightedSum, SafetyTensorOps<T>.Where(hasMagnitude, magSum, 1.0)),
                NumOps.FromDouble((double)sampleRate / frameSize)),
            0.0);

        // Pitch estimation (autocorrelation), kept only when positive.
        var pitch = SafetyTensorOps<T>.AutocorrelationPitch(frames, sampleRate, Math.Min(sampleRate / 50, frameSize / 2));
        return (energy, zcr, centroid, pitch, voiced);
    }

    private static Tensor<T> Join(List<Tensor<T>> parts)
        => parts.Count == 1 ? parts[0] : SafetyTensorOps<T>.Engine.TensorConcatenate(parts.ToArray(), 0);

    private static T ComputeEnergyAggressionScore(Tensor<T> energies)
    {
        var (mean, variance) = SafetyTensorOps<T>.MeanVariance(energies);

        // High energy + high variance = aggressive
        // RMS energy for speech typically ~0.01-0.05; shouting ~0.1-0.5
        T rms = NumOps.Sqrt(mean);
        T energyIndicator = SafetyTensorOps<T>.MinOne(NumOps.Divide(rms, NumOps.FromDouble(0.3)));

        // Energy variance (dynamic range)
        T cv = NumOps.GreaterThan(mean, NumOps.FromDouble(1e-10))
            ? NumOps.Divide(NumOps.Sqrt(variance), mean)
            : NumOps.Zero;
        T dynamicIndicator = SafetyTensorOps<T>.MinOne(NumOps.Divide(cv, NumOps.FromDouble(2.0)));

        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(0.6), energyIndicator),
            NumOps.Multiply(NumOps.FromDouble(0.4), dynamicIndicator));
    }

    private static T ComputePitchAggressionScore(Tensor<T> pitches)
    {
        if (pitches.Length < 2) return NumOps.Zero;

        var (mean, variance) = SafetyTensorOps<T>.MeanVariance(pitches);

        // High pitch mean (>200 Hz) indicates agitation
        // Normal conversational: 100-150 Hz (male), 180-250 Hz (female)
        // Aggressive: often > 200 Hz (male), > 300 Hz (female)
        T pitchElevation = SafetyTensorOps<T>.ClampUnit(
            NumOps.Divide(NumOps.Subtract(mean, NumOps.FromDouble(180)), NumOps.FromDouble(200)));

        // Pitch variance (rapid changes indicate anger)
        T pitchVariability = SafetyTensorOps<T>.MinOne(NumOps.Divide(NumOps.Sqrt(variance), NumOps.FromDouble(100)));

        T half = NumOps.FromDouble(0.5);
        return NumOps.Add(NumOps.Multiply(half, pitchElevation), NumOps.Multiply(half, pitchVariability));
    }

    private static T ComputeSpectralAggressionScore(Tensor<T> centroids)
    {
        if (centroids.Length == 0) return NumOps.Zero;

        T mean = SafetyTensorOps<T>.Engine.TensorMean(centroids);

        // High spectral centroid = harsh, aggressive timbre
        // Normal speech: ~1000-2000 Hz; shouting: ~2000-4000 Hz
        return SafetyTensorOps<T>.ClampUnit(NumOps.Divide(NumOps.Subtract(mean, NumOps.FromDouble(1500)), NumOps.FromDouble(2500)));
    }

    private static T ComputeSpeechRateScore(Tensor<T> zcrs)
    {
        if (zcrs.Length == 0) return NumOps.Zero;

        T mean = SafetyTensorOps<T>.Engine.TensorMean(zcrs);

        // High ZCR indicates fast articulation
        // Normal: ~0.05-0.1; fast/aggressive: ~0.15-0.3
        return SafetyTensorOps<T>.ClampUnit(NumOps.Divide(NumOps.Subtract(mean, NumOps.FromDouble(0.1)), NumOps.FromDouble(0.2)));
    }
}
