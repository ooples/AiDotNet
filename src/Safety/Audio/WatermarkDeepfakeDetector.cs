using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AudioSeal: Proactive Detection of Voice Cloning with Localized Watermarking",
    "https://arxiv.org/abs/2401.17264",
    Year = 2024,
    Authors = "Robin San Roman, Pierre Fernandez, et al.")]
public class WatermarkDeepfakeDetector<T> : AudioSafetyModuleBase<T>
{

    private readonly T _threshold;

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
        var audio = SafetyTensorOps<T>.AsTensor(audioSamples);

        // Check for various watermark signatures
        T spreadSpectrumScore = DetectSpreadSpectrumWatermark(audio, sampleRate);
        T subBandScore = DetectSubBandWatermark(audio, sampleRate);
        T phaseScore = DetectPhaseWatermark(audio, sampleRate);

        // Also check for natural recording artifacts (their absence is suspicious)
        T naturalArtifactScore = DetectNaturalRecordingArtifacts(audio, sampleRate);

        // Combined watermark detection
        T watermarkScoreT = SafetyTensorOps<T>.MaxNaN(spreadSpectrumScore,
            SafetyTensorOps<T>.MaxNaN(subBandScore, phaseScore));
        double watermarkScore = NumOps.ToDouble(watermarkScoreT);

        if (NumOps.GreaterThanOrEquals(watermarkScoreT, _threshold))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, watermarkScore),
                Description = $"Audio watermark detected (score: {watermarkScore:F3}). " +
                              $"Spread-spectrum: {NumOps.ToDouble(spreadSpectrumScore):F3}, sub-band: {NumOps.ToDouble(subBandScore):F3}, " +
                              $"phase: {NumOps.ToDouble(phaseScore):F3}. This indicates AI-generated audio " +
                              $"from a responsible provider.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        // If no watermark but also no natural artifacts → suspicious
        T suspiciousScoreT = NumOps.Multiply(NumOps.Subtract(NumOps.One, watermarkScoreT),
            NumOps.Subtract(NumOps.One, naturalArtifactScore));
        if (NumOps.GreaterThanOrEquals(suspiciousScoreT, _threshold))
        {
            double suspiciousScore = NumOps.ToDouble(suspiciousScoreT);
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

    private static T DetectSpreadSpectrumWatermark(Tensor<T> audio, int sampleRate)
    {
        var engine = SafetyTensorOps<T>.Engine;

        // Spread-spectrum watermarks distribute energy uniformly across frequencies
        int frameSize = 512;
        int numFrames = Math.Min(8, audio.Length / frameSize);
        if (numFrames < 2) return NumOps.Zero;

        // Check for unusually uniform mid-band energy across frames
        var frames = SafetyTensorOps<T>.Frames(audio, frameSize, frameSize, numFrames);
        var magnitudes = SafetyTensorOps<T>.FftMagnitude(frames);
        int halfLen = frameSize / 2;
        int lowBand = halfLen / 4;
        int highBand = halfLen * 3 / 4;

        // Spectral flatness in mid-band over the bins with magnitude > 1e-20
        var axis1 = new[] { 1 };
        var band = engine.TensorNarrow(magnitudes, 1, lowBand, highBand - lowBand);
        var present = SafetyTensorOps<T>.Gt(band, 1e-20);
        var logSum = engine.ReduceSum(
            SafetyTensorOps<T>.Where(present, engine.TensorLog(SafetyTensorOps<T>.Where(present, band, 1.0)), 0.0),
            axis1, false);
        var arithSum = engine.ReduceSum(SafetyTensorOps<T>.Where(present, band, 0.0), axis1, false);
        var count = engine.ReduceSum(present, axis1, false);

        var valid = SafetyTensorOps<T>.And(SafetyTensorOps<T>.Gt(count, 0.0), SafetyTensorOps<T>.Gt(arithSum, 1e-20));
        var safeCount = SafetyTensorOps<T>.Where(valid, count, 1.0);
        var geoMean = engine.TensorExp(engine.TensorDivide(logSum, safeCount));
        var arithMean = engine.TensorDivide(SafetyTensorOps<T>.Where(valid, arithSum, 1.0), safeCount);
        var midBandFlatness = SafetyTensorOps<T>.Compact(engine.TensorDivide(geoMean, arithMean), valid);

        if (midBandFlatness.Length < 2) return NumOps.Zero;

        // High and consistent mid-band flatness = spread-spectrum watermark
        var (meanFlatness, variance) = SafetyTensorOps<T>.MeanVariance(midBandFlatness);

        // Watermarked: mean flatness > 0.6, low variance
        T flatnessScore = SafetyTensorOps<T>.MaxZero(
            NumOps.Multiply(NumOps.Subtract(meanFlatness, NumOps.FromDouble(0.5)), NumOps.FromDouble(2)));
        T consistencyScore = SafetyTensorOps<T>.MaxZero(
            NumOps.Subtract(NumOps.One, NumOps.Multiply(variance, NumOps.FromDouble(50))));

        return SafetyTensorOps<T>.MinOne(NumOps.Multiply(flatnessScore, consistencyScore));
    }

    private static T DetectSubBandWatermark(Tensor<T> audio, int sampleRate)
    {
        var engine = SafetyTensorOps<T>.Engine;

        // Sub-band watermarks embed energy in specific narrow frequency bands
        int frameSize = 1024;
        if (audio.Length < frameSize) return NumOps.Zero;

        int mid = (audio.Length - frameSize) / 2;
        var frame = engine.Reshape(engine.TensorNarrow(audio, 0, mid, frameSize), new[] { 1, frameSize });

        // Compute magnitude spectrum
        int halfLen = frameSize / 2;
        var magnitudes = engine.Reshape(
            engine.TensorNarrow(SafetyTensorOps<T>.FftMagnitude(frame), 1, 0, halfLen), new[] { halfLen });

        // Peak test at every bin k in [2, halfLen-2]:
        // m[k] > m[k-1] && m[k] > m[k+1] && m[k] > m[k-2] * 1.5
        int inner = halfLen - 3;
        var center = engine.TensorNarrow(magnitudes, 0, 2, inner);
        var isPeak = SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.And(
                engine.TensorGreaterThan(center, engine.TensorNarrow(magnitudes, 0, 1, inner)),
                engine.TensorGreaterThan(center, engine.TensorNarrow(magnitudes, 0, 3, inner))),
            engine.TensorGreaterThan(center,
                engine.TensorMultiplyScalar(engine.TensorNarrow(magnitudes, 0, 0, inner), NumOps.FromDouble(1.5))));
        var peaks = engine.TensorConcatenate(new[]
        {
            Tensor<T>.CreateZeros(2), isPeak, Tensor<T>.CreateZeros(1)
        }, 0);

        // Look for unusually regular energy peaks at specific intervals
        // (watermarks often use fixed frequency spacing): for each spacing s in [4, 32],
        // count the peaks at k = s, 2s, ... while k < halfLen - s.
        int spacings = 32 - 4 + 1;
        var k = engine.Reshape(SafetyTensorOps<T>.Range(halfLen), new[] { 1, halfLen });
        var s = engine.Reshape(engine.TensorAddScalar(SafetyTensorOps<T>.Range(spacings), NumOps.FromDouble(4)),
            new[] { spacings, 1 });
        var grid = new[] { spacings, halfLen };
        var kGrid = engine.TensorBroadcastTo(k, grid);
        var sGrid = engine.TensorBroadcastTo(s, grid);
        var onMultiple = engine.TensorEquals(engine.TensorRemainder(kGrid, sGrid), NumOps.Zero);
        var inRange = SafetyTensorOps<T>.And(
            engine.TensorGreaterEqual(kGrid, sGrid),
            engine.TensorLessThan(kGrid, engine.ScalarMinusTensor(NumOps.FromDouble(halfLen), sGrid)));
        var selection = SafetyTensorOps<T>.And(onMultiple, inRange);
        var peakCounts = engine.TensorMatMul(selection, engine.Reshape(peaks, new[] { halfLen, 1 }));
        T maxPeakCount = SafetyTensorOps<T>.MaxNaN(NumOps.Zero, engine.TensorMaxValue(peakCounts));

        int expectedPeaks = halfLen / 16; // Expected for watermarked audio
        return SafetyTensorOps<T>.MinOne(NumOps.Divide(maxPeakCount, NumOps.FromDouble(expectedPeaks)));
    }

    private static T DetectPhaseWatermark(Tensor<T> audio, int sampleRate)
    {
        var engine = SafetyTensorOps<T>.Engine;

        // Phase-based watermarks encode data in phase differences between frames
        int frameSize = 512;
        int numFrames = Math.Min(4, audio.Length / frameSize);
        if (numFrames < 2) return NumOps.Zero;

        var frames = SafetyTensorOps<T>.Frames(audio, frameSize, frameSize, numFrames);
        engine.FFT(frames, Tensor<T>.CreateZeros(numFrames, frameSize), out var re, out var im);
        int halfLen = frameSize / 2;
        var phases = engine.TensorNarrow(engine.TensorAtan2(im, re), 1, 0, halfLen);

        // Check if phase differences between consecutive frames cluster around discrete values
        // (watermarks encode bits as specific phase offsets), bins 10 .. halfLen-1
        var band = engine.TensorNarrow(phases, 1, 10, halfLen - 10);
        var diff = engine.TensorDiff(band, 1, 0);

        // Normalize to [-pi, pi]
        diff = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Gt(diff, Math.PI),
            engine.TensorSubtractScalar(diff, NumOps.FromDouble(2 * Math.PI)), diff);
        diff = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Lt(diff, -Math.PI),
            engine.TensorAddScalar(diff, NumOps.FromDouble(2 * Math.PI)), diff);

        // Check if close to 0, pi/2, pi, or -pi/2
        var absDiff = engine.TensorAbs(diff);
        var discrete = SafetyTensorOps<T>.Or(
            SafetyTensorOps<T>.Or(
                SafetyTensorOps<T>.Lt(absDiff, 0.1),
                SafetyTensorOps<T>.Lt(engine.TensorAbs(engine.TensorSubtractScalar(absDiff, NumOps.FromDouble(Math.PI / 2))), 0.1)),
            SafetyTensorOps<T>.Lt(engine.TensorAbs(engine.TensorSubtractScalar(absDiff, NumOps.FromDouble(Math.PI))), 0.1));
        var discreteRatio = engine.TensorDivideScalar(
            engine.ReduceSum(discrete, new[] { 1 }, false), NumOps.FromDouble(halfLen));

        T maxConsistency = SafetyTensorOps<T>.FirstStrictMax(
            engine.Reshape(discreteRatio, new[] { 1, numFrames - 1 }), 0.0).Best.GetFlat(0);

        // High ratio of discrete phase differences = likely watermarked
        return SafetyTensorOps<T>.ClampUnit(
            NumOps.Multiply(NumOps.Subtract(maxConsistency, NumOps.FromDouble(0.3)), NumOps.FromDouble(2)));
    }

    private static T DetectNaturalRecordingArtifacts(Tensor<T> audio, int sampleRate)
    {
        var engine = SafetyTensorOps<T>.Engine;

        // Natural recordings have: background noise, slight DC offset, frequency roll-off
        int frameSize = Math.Min(1024, audio.Length);
        var frame = engine.TensorNarrow(audio, 0, 0, frameSize);

        // DC offset (natural recordings often have slight DC bias)
        T dcOffset = NumOps.Abs(NumOps.Divide(engine.TensorSum(frame), NumOps.FromDouble(frameSize)));
        T hasDC = NumOps.GreaterThan(dcOffset, NumOps.FromDouble(0.001)) ? NumOps.One : NumOps.Zero;

        // Background noise floor (silent portions should have some noise): 10th percentile of
        // the sorted magnitudes. NaN sorts first, so it is placed below every magnitude (-1).
        var magnitude = engine.TensorAbs(frame);
        var ordered = engine.TensorWhere(engine.TensorIsNan(magnitude),
            SafetyTensorOps<T>.FullLike(magnitude, -1.0), magnitude);
        var sorted = engine.TensorSort(ordered, 0, false).Values;
        T noiseFloor = sorted.GetFlat(frameSize / 10);
        T hasNoise = NumOps.GreaterThan(noiseFloor, NumOps.FromDouble(0.0001)) ? NumOps.One : NumOps.Zero;

        // Natural = has DC + has noise → score ~1.0
        return NumOps.Divide(NumOps.Add(hasDC, hasNoise), NumOps.FromDouble(2.0));
    }
}
