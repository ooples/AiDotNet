using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Protects voice recordings against cloning by adding imperceptible adversarial perturbations.
/// </summary>
/// <remarks>
/// <para>
/// Adds carefully crafted perturbations to the audio that disrupt voice cloning systems while
/// remaining inaudible to human listeners. The perturbations target the frequency bands most
/// important for speaker embedding extraction (typically 300-3400 Hz for formants) and add
/// noise shaped to psychoacoustic masking thresholds.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like adding an invisible "anti-copying" pattern to a voice
/// recording. Humans can't hear the difference, but if someone tries to clone the voice using
/// AI, the protection disrupts the cloning process. Think of it like a DRM for your voice.
/// </para>
/// <para>
/// <b>References:</b>
/// - SafeSpeech: SPEC perturbation framework against voice cloning (2025, arxiv:2504.09839)
/// - Adversarial examples for speech recognition (Carlini &amp; Wagner, 2018)
/// - Voice protection via perturbation (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis",
    "https://arxiv.org/abs/2504.09839",
    Year = 2025,
    Authors = "Various")]
public class PerturbationVoiceProtector<T> : AudioSafetyModuleBase<T>
{

    private readonly T _perturbationStrength;

    /// <inheritdoc />
    public override string ModuleName => "PerturbationVoiceProtector";

    /// <summary>
    /// Initializes a new perturbation-based voice protector.
    /// </summary>
    /// <param name="perturbationStrength">
    /// Perturbation strength (0-1). Default: 0.02. Higher values provide more protection
    /// but may become audible.
    /// </param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public PerturbationVoiceProtector(double perturbationStrength = 0.02, int sampleRate = 16000)
        : base(sampleRate)
    {
        if (perturbationStrength <= 0 || perturbationStrength > 1)
            throw new ArgumentOutOfRangeException(nameof(perturbationStrength), perturbationStrength,
                "Perturbation strength must be in (0, 1].");
        _perturbationStrength = NumOps.FromDouble(perturbationStrength);
    }

    /// <summary>
    /// Applies voice protection perturbations and returns protected audio with safety findings.
    /// </summary>
    public Vector<T> ProtectAudio(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples.Length < 512)
        {
            return audioSamples;
        }

        using var noGrad = new NoGradScope<T>();
        var engine = SafetyTensorOps<T>.Engine;
        int frameSize = 512;
        int hopSize = 256;
        int length = audioSamples.Length;
        int numFrames = (length - frameSize) / hopSize + 1;
        var audio = SafetyTensorOps<T>.AsTensor(audioSamples);

        // Process in overlapping frames: [numFrames, frameSize]
        var frames = SafetyTensorOps<T>.Frames(audio, frameSize, hopSize, numFrames);

        // FFT
        SafetyTensorOps<T>.Dft(frames, out var real, out var imag);
        int halfLen = frameSize / 2;

        // Add perturbation in formant regions (300-3400 Hz)
        int lowBin = (int)(300.0 * frameSize / sampleRate);
        int highBin = (int)(3400.0 * frameSize / sampleRate);
        highBin = Math.Min(highBin, halfLen - 1);

        if (highBin >= lowBin)
        {
            int bins = highBin - lowBin + 1;
            var bandReal = engine.TensorNarrow(real, 1, lowBin, bins);
            var bandImag = engine.TensorNarrow(imag, 1, lowBin, bins);
            var magnitude = engine.TensorSqrt(engine.TensorAdd(engine.TensorSquare(bandReal), engine.TensorSquare(bandImag)));

            // Hash-based pseudo-random perturbation (deterministic for reproducibility),
            // proportional to local magnitude (masking-aware)
            var hash = SafetyTensorOps<T>.HashInt(SafetyTensorOps<T>.HashKeys(numFrames, hopSize * 1000, lowBin, bins));
            var pertPhase = engine.TensorMultiplyScalar(SafetyTensorOps<T>.ModuloAsT(hash, 1000),
                NumOps.FromDouble(2.0 * Math.PI / 1000.0)); // Random phase [0, 2*pi)
            var pertMag = engine.TensorMultiply(
                engine.TensorMultiplyScalar(magnitude, _perturbationStrength),
                engine.TensorDivideScalar(SafetyTensorOps<T>.ModuloAsT(hash, 100), NumOps.FromDouble(100.0)));

            var left = Tensor<T>.CreateZeros(numFrames, lowBin);
            var right = Tensor<T>.CreateZeros(numFrames, frameSize - lowBin - bins);
            real = engine.TensorAdd(real, engine.TensorConcatenate(new[]
            {
                left, engine.TensorMultiply(pertMag, engine.TensorCos(pertPhase)), right
            }, 1));
            imag = engine.TensorAdd(imag, engine.TensorConcatenate(new[]
            {
                left, engine.TensorMultiply(pertMag, engine.TensorSin(pertPhase)), right
            }, 1));
        }

        // Inverse FFT
        engine.IFFT(real, imag, out var perturbedFrames, out _);

        // Overlap-add with Hann window: protected += w * perturbed - w * frame
        var window = SafetyTensorOps<T>.SymmetricHann(frameSize);
        var delta = engine.TensorSubtract(
            engine.TensorMultiply(perturbedFrames, window), engine.TensorMultiply(frames, window));
        var (current, previous) = SafetyTensorOps<T>.HalfOverlapParts(delta, length);

        // The earlier frame's contribution lands first, as in the sequential overlap-add.
        return engine.TensorAdd(engine.TensorAdd(audio, previous), current).ToVector();
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        // This module is primarily a protection tool, but can detect if audio
        // already has perturbation-like artifacts
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 1024) return findings;

        // Check for existing perturbation artifacts in formant band
        T artifactScoreT = DetectPerturbationArtifacts(audioSamples, sampleRate);

        if (NumOps.GreaterThanOrEquals(artifactScoreT, NumOps.FromDouble(0.5)))
        {
            double artifactScore = NumOps.ToDouble(artifactScoreT);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.AIGenerated,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, artifactScore),
                Description = $"Voice perturbation artifacts detected (score: {artifactScore:F3}). " +
                              $"Audio may have been processed with anti-cloning protection.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static T DetectPerturbationArtifacts(Vector<T> audio, int sampleRate)
    {
        int frameSize = 512;
        if (audio.Length < frameSize * 2) return NumOps.Zero;

        using var noGrad = new NoGradScope<T>();
        var engine = SafetyTensorOps<T>.Engine;

        // Compare adjacent frames for unnatural spectral discontinuities in formant band
        int numPairs = Math.Min(4, audio.Length / frameSize - 1);
        var frames = SafetyTensorOps<T>.Frames(SafetyTensorOps<T>.AsTensor(audio), frameSize, frameSize, numPairs + 1);
        var magnitudes = SafetyTensorOps<T>.FftMagnitude(frames);
        int halfLen = frameSize / 2;

        int lowBin = (int)(300.0 * frameSize / sampleRate);
        int highBin = Math.Min((int)(3400.0 * frameSize / sampleRate), halfLen - 1);

        T anomalySum = NumOps.Zero;
        if (highBin >= lowBin)
        {
            var band = engine.TensorNarrow(magnitudes, 1, lowBin, highBin - lowBin + 1);
            var m1 = engine.TensorNarrow(band, 0, 0, numPairs);
            var m2 = engine.TensorNarrow(band, 0, 1, numPairs);
            var axis1 = new[] { 1 };
            var diffSum = engine.ReduceSum(engine.TensorAbs(engine.TensorSubtract(m1, m2)), axis1, false);
            var magSum = engine.ReduceSum(
                engine.TensorDivideScalar(engine.TensorAdd(m1, m2), NumOps.FromDouble(2)), axis1, false);
            var valid = SafetyTensorOps<T>.Gt(magSum, 1e-10);
            var relDiff = SafetyTensorOps<T>.Where(valid,
                engine.TensorDivide(diffSum, SafetyTensorOps<T>.Where(valid, magSum, 1.0)), 0.0);
            anomalySum = engine.TensorSum(relDiff);
        }

        T meanAnomaly = NumOps.Divide(anomalySum, NumOps.FromDouble(numPairs));
        return SafetyTensorOps<T>.ClampUnit(
            NumOps.Multiply(NumOps.Subtract(meanAnomaly, NumOps.FromDouble(0.3)), NumOps.FromDouble(3)));
    }
}
