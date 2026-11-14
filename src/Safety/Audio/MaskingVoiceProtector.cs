using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Protects voice recordings against cloning using psychoacoustic masking — adding noise
/// that is hidden beneath audible content but disrupts speaker embedding extraction.
/// </summary>
/// <remarks>
/// <para>
/// Applies frequency-domain noise shaped to the psychoacoustic masking curve of the audio.
/// The masking curve determines the threshold below which noise is inaudible to humans;
/// this protector adds maximum disruption noise just below this threshold. The disruption
/// specifically targets the frequency bands used by speaker verification systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> Our ears have blind spots — when a loud sound is playing, we can't
/// hear quiet sounds nearby. This module exploits those blind spots to hide anti-cloning noise
/// where your ears can't detect it, but AI cloning systems can "hear" and get confused by.
/// </para>
/// <para>
/// <b>References:</b>
/// - VocalCrypt: Pseudo-timbre jamming for voice protection (2025, arxiv:2502.10329)
/// - Psychoacoustic masking models for audio steganography (2023)
/// - MPEG psychoacoustic model for perceptual coding (ISO/IEC 11172-3)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("VocalCrypt: Novel Active Defense Against Deepfake Voice Based on Masking Effect",
    "https://arxiv.org/abs/2502.10329",
    Year = 2025,
    Authors = "Various")]
public class MaskingVoiceProtector<T> : AudioSafetyModuleBase<T>
{

    private readonly T _maskingStrength;

    /// <inheritdoc />
    public override string ModuleName => "MaskingVoiceProtector";

    /// <summary>
    /// Initializes a new masking-based voice protector.
    /// </summary>
    /// <param name="maskingStrength">
    /// Masking noise strength relative to masking threshold (0-1). Default: 0.8.
    /// 1.0 = add noise at exactly the masking threshold; 0.5 = half threshold.
    /// </param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public MaskingVoiceProtector(double maskingStrength = 0.8, int sampleRate = 16000)
        : base(sampleRate)
    {
        _maskingStrength = NumOps.FromDouble(maskingStrength);
    }

    /// <summary>
    /// Applies psychoacoustic masking protection and returns protected audio.
    /// </summary>
    public Vector<T> ProtectAudio(Vector<T> audioSamples, int sampleRate)
    {
        if (audioSamples.Length < 512) return audioSamples;

        using var noGrad = new NoGradScope<T>();
        var engine = SafetyTensorOps<T>.Engine;
        int frameSize = 512;
        int hopSize = 256;
        int length = audioSamples.Length;
        int numFrames = (length - frameSize) / hopSize + 1;
        var audio = SafetyTensorOps<T>.AsTensor(audioSamples);

        // Window every frame: [numFrames, frameSize]
        var window = SafetyTensorOps<T>.SymmetricHann(frameSize);
        var frames = engine.TensorMultiply(SafetyTensorOps<T>.Frames(audio, frameSize, hopSize, numFrames), window);

        SafetyTensorOps<T>.Dft(frames, out var real, out var imag);
        int halfLen = frameSize / 2;

        // Compute magnitude spectrum and the simplified masking curve
        var magnitudes = engine.TensorNarrow(
            engine.TensorSqrt(engine.TensorAdd(engine.TensorSquare(real), engine.TensorSquare(imag))), 1, 0, halfLen);
        var maskingThresholds = ComputeMaskingCurve(magnitudes, halfLen, sampleRate, frameSize);

        // Add masking noise in speaker-embedding-critical bands (bins 1 .. halfLen-1).
        // Focus on formant region (300-3400 Hz) and speaker-characteristic region (1-6 kHz)
        int bins = halfLen - 1;
        var freq = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(SafetyTensorOps<T>.Range(bins), NumOps.One),
            NumOps.FromDouble((double)sampleRate / frameSize));
        var outside = SafetyTensorOps<T>.Or(SafetyTensorOps<T>.Lt(freq, 300), SafetyTensorOps<T>.Gt(freq, 6000));
        var core = SafetyTensorOps<T>.And(SafetyTensorOps<T>.Ge(freq, 1000), SafetyTensorOps<T>.Le(freq, 4000));
        var bandWeight = SafetyTensorOps<T>.Where(outside, 0.2, SafetyTensorOps<T>.Where(core, 1.0, 0.6));

        // Noise amplitude = masking threshold * strength * band weight
        var noiseAmp = engine.TensorMultiply(
            engine.TensorMultiplyScalar(engine.TensorNarrow(maskingThresholds, 1, 1, bins), _maskingStrength),
            bandWeight);

        // Pseudo-random phase from the hash of (start * 1000 + k)
        var hash = SafetyTensorOps<T>.HashInt(SafetyTensorOps<T>.HashKeys(numFrames, hopSize * 1000, 1, bins));
        var phase = engine.TensorDivideScalar(SafetyTensorOps<T>.ModuloAsT(hash, 6283), NumOps.FromDouble(1000.0));
        var noiseReal = engine.TensorMultiply(noiseAmp, engine.TensorCos(phase));
        var noiseImag = engine.TensorMultiply(noiseAmp, engine.TensorSin(phase));

        // Only the positive-frequency bins 1 .. halfLen-1 receive noise.
        var dc = Tensor<T>.CreateZeros(numFrames, 1);
        var upper = Tensor<T>.CreateZeros(numFrames, frameSize - halfLen);
        real = engine.TensorAdd(real, engine.TensorConcatenate(new[] { dc, noiseReal, upper }, 1));
        imag = engine.TensorAdd(imag, engine.TensorConcatenate(new[] { dc, noiseImag, upper }, 1));

        // Inverse transform (real part), then window for overlap-add
        engine.IFFT(real, imag, out var protectedFrames, out _);
        var weighted = engine.TensorMultiply(protectedFrames, window);
        var windowEnergy = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorSquare(window), new[] { 1, frameSize }), new[] { numFrames, frameSize });

        // Overlap-add
        var (current, previous) = SafetyTensorOps<T>.HalfOverlapParts(weighted, length);
        var result = engine.TensorAdd(previous, current);
        var (currentWeight, previousWeight) = SafetyTensorOps<T>.HalfOverlapParts(windowEnergy, length);
        var windowSum = engine.TensorAdd(previousWeight, currentWeight);

        // Normalize
        var covered = SafetyTensorOps<T>.Gt(windowSum, 1e-10);
        var normalized = engine.TensorDivide(result, SafetyTensorOps<T>.Where(covered, windowSum, 1.0));
        return SafetyTensorOps<T>.Where(covered, normalized, audio).ToVector();
    }

    /// <summary>
    /// Computes a simplified psychoacoustic masking curve.
    /// Based on the spreading function model from MPEG psychoacoustic analysis.
    /// </summary>
    private static Tensor<T> ComputeMaskingCurve(Tensor<T> magnitudes, int halfLen,
        int sampleRate, int frameSize)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int frameCount = magnitudes.Shape[0];

        // Convert to dB scale
        var audible = SafetyTensorOps<T>.Gt(magnitudes, 1e-20);
        var dB = SafetyTensorOps<T>.Where(audible,
            engine.TensorMultiplyScalar(
                engine.TensorLog(SafetyTensorOps<T>.Where(audible, magnitudes, 1.0)),
                NumOps.FromDouble(20.0 / Math.Log(10.0))),
            -100.0);

        // Simultaneous masking: each spectral peak masks nearby frequencies.
        // maxMask[k] = max(-100, max over j in [k - spread, k + spread) of dB[j] - 2|k - j|);
        // the padding (-1000) never beats the -100 floor.
        int spreadRange = Math.Min(32, halfLen / 4);
        var pad = SafetyTensorOps<T>.Full(new[] { frameCount, spreadRange }, NumOps.FromDouble(-1000.0));
        var padded = engine.TensorConcatenate(new[] { pad, dB, pad }, 1);
        var maxMask = SafetyTensorOps<T>.Full(new[] { frameCount, halfLen }, NumOps.FromDouble(-100.0));
        // Simplified spreading function: -10 dB per critical band (2 dB per bin of distance)
        var penalty = engine.TensorMultiplyScalar(
            engine.TensorAbs(engine.TensorArange<T>(-spreadRange, spreadRange, 1)), NumOps.FromDouble(2.0));
        for (int offset = -spreadRange; offset < spreadRange; offset++)
        {
            var neighbour = engine.TensorNarrow(padded, 1, spreadRange + offset, halfLen);
            var spreading = engine.TensorSubtract(neighbour, engine.TensorNarrow(penalty, 0, spreadRange + offset, 1));
            maxMask = engine.TensorMax(maxMask, spreading);
        }

        // Absolute hearing threshold (simplified) — varies by frequency
        var freq = engine.TensorMultiplyScalar(SafetyTensorOps<T>.Range(halfLen),
            NumOps.FromDouble((double)sampleRate / frameSize));
        var absThreshold = engine.TensorBroadcastTo(
            engine.Reshape(ComputeAbsoluteThreshold(freq), new[] { 1, halfLen }), new[] { frameCount, halfLen });

        // Masking threshold = max of simultaneous masking and absolute threshold
        // Subtract offset (masking is ~15 dB below masker for tonal, ~5 dB for noise)
        T maskingOffset = NumOps.FromDouble(10.0); // dB below masker
        var maskedThreshold = engine.TensorMax(engine.TensorSubtractScalar(maxMask, maskingOffset), absThreshold);

        // Convert back to linear: 10^(dB / 20)
        return engine.TensorExp(engine.TensorMultiplyScalar(maskedThreshold, NumOps.FromDouble(Math.Log(10.0) / 20.0)));
    }

    /// <summary>
    /// Simplified absolute hearing threshold (ISO 226).
    /// </summary>
    private static Tensor<T> ComputeAbsoluteThreshold(Tensor<T> freq)
    {
        var engine = SafetyTensorOps<T>.Engine;
        var outOfRange = SafetyTensorOps<T>.Or(SafetyTensorOps<T>.Lt(freq, 20), SafetyTensorOps<T>.Gt(freq, 20000));

        // Simplified approximation of the minimum audibility curve
        var f = engine.TensorDivideScalar(SafetyTensorOps<T>.Where(outOfRange, 1000.0, freq), NumOps.FromDouble(1000.0)); // kHz
        var shifted = engine.TensorSubtractScalar(f, NumOps.FromDouble(3.3));
        var threshold = engine.TensorAdd(
            engine.TensorSubtract(
                engine.TensorMultiplyScalar(engine.TensorPower(f, NumOps.FromDouble(-0.8)), NumOps.FromDouble(3.64)),
                engine.TensorMultiplyScalar(
                    engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(shifted), NumOps.FromDouble(-0.6))),
                    NumOps.FromDouble(6.5))),
            engine.TensorMultiplyScalar(engine.TensorPower(f, NumOps.FromDouble(4)), NumOps.FromDouble(0.001)));

        var clamped = engine.TensorMax(engine.TensorMin(threshold, NumOps.FromDouble(80)), NumOps.FromDouble(-60));
        return SafetyTensorOps<T>.Where(outOfRange, 80.0, clamped);
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        // Detect if audio has psychoacoustic masking-based protection
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 1024) return findings;

        // Look for noise below masking threshold (unusual for natural audio)
        T protectionScoreT = DetectMaskingProtection(audioSamples, sampleRate);

        if (NumOps.GreaterThanOrEquals(protectionScoreT, NumOps.FromDouble(0.5)))
        {
            double protectionScore = NumOps.ToDouble(protectionScoreT);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.AIGenerated,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, protectionScore),
                Description = $"Psychoacoustic masking protection detected (score: {protectionScore:F3}). " +
                              $"Audio appears to have voice protection applied.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static T DetectMaskingProtection(Vector<T> audio, int sampleRate)
    {
        int frameSize = 512;
        if (audio.Length < frameSize) return NumOps.Zero;

        using var noGrad = new NoGradScope<T>();
        var engine = SafetyTensorOps<T>.Engine;

        // Analyze noise floor in typically quiet frequency regions
        int mid = (audio.Length - frameSize) / 2;
        var frame = engine.TensorMultiply(
            engine.Reshape(engine.TensorNarrow(SafetyTensorOps<T>.AsTensor(audio), 0, mid, frameSize), new[] { 1, frameSize }),
            SafetyTensorOps<T>.SymmetricHann(frameSize));

        int halfLen = frameSize / 2;
        var magnitudes = engine.TensorNarrow(SafetyTensorOps<T>.FftMagnitude(frame), 1, 0, halfLen);

        // Check if energy in masking-threshold-level regions is higher than expected (bins 5 ..)
        var maskCurve = ComputeMaskingCurve(magnitudes, halfLen, sampleRate, frameSize);
        var curve = engine.TensorNarrow(maskCurve, 1, 5, halfLen - 5);
        var mags = engine.TensorNarrow(magnitudes, 1, 5, halfLen - 5);
        var counted = SafetyTensorOps<T>.Gt(curve, 1e-10);
        var ratio = engine.TensorDivide(mags, SafetyTensorOps<T>.Where(counted, curve, 1.0));
        var near = SafetyTensorOps<T>.And(counted,
            SafetyTensorOps<T>.And(SafetyTensorOps<T>.Gt(ratio, 0.5), SafetyTensorOps<T>.Lt(ratio, 2.0)));
        T nearThresholdBins = engine.TensorSum(near);
        T totalBins = engine.TensorSum(counted);

        return NumOps.GreaterThan(totalBins, NumOps.Zero)
            ? SafetyTensorOps<T>.MinOne(NumOps.Multiply(NumOps.Divide(nearThresholdBins, totalBins), NumOps.FromDouble(3)))
            : NumOps.Zero;
    }
}
