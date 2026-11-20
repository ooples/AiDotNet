using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Protects voice recordings by embedding imperceptible watermarks that survive voice cloning
/// and can be detected in cloned output.
/// </summary>
/// <remarks>
/// <para>
/// Embeds a spread-spectrum watermark into the audio's mid-frequency band that is robust to
/// typical voice processing (resampling, compression, noise addition) and voice cloning.
/// The watermark can later be detected using <see cref="WatermarkDeepfakeDetector{T}"/>.
/// Uses frequency-domain embedding with psychoacoustic masking to ensure inaudibility.
/// </para>
/// <para>
/// <b>For Beginners:</b> This embeds a secret "tag" in the audio that follows the voice
/// even if someone tries to clone it. If cloned audio shows up later, the tag proves where
/// the original voice came from.
/// </para>
/// <para>
/// <b>References:</b>
/// - AudioSeal: Localized watermarking for speech (Meta AI, 2024, arxiv:2401.17264)
/// - SoK: Systematization of watermarking across modalities (2024, arxiv:2411.18479)
/// - Watermarking survey: unified framework (2025, arxiv:2504.03765)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AudioSeal: Proactive Detection of Voice Cloning with Localized Watermarking",
    "https://arxiv.org/abs/2401.17264",
    Year = 2024,
    Authors = "Robin San Roman, Pierre Fernandez, et al.")]
public class WatermarkVoiceProtector<T> : AudioSafetyModuleBase<T>
{

    private readonly T _watermarkStrength;
    private readonly int _watermarkKey;

    /// <inheritdoc />
    public override string ModuleName => "WatermarkVoiceProtector";

    /// <summary>
    /// Initializes a new watermark-based voice protector.
    /// </summary>
    /// <param name="watermarkStrength">
    /// Watermark strength (0-1). Default: 0.01. Trade-off between detectability and audibility.
    /// </param>
    /// <param name="watermarkKey">
    /// Secret key for watermark generation. Default: 42. Different keys produce different watermarks.
    /// </param>
    /// <param name="sampleRate">Default sample rate in Hz. Default: 16000.</param>
    public WatermarkVoiceProtector(double watermarkStrength = 0.01, int watermarkKey = 42, int sampleRate = 16000)
        : base(sampleRate)
    {
        _watermarkStrength = NumOps.FromDouble(watermarkStrength);
        _watermarkKey = watermarkKey;
    }

    /// <summary>
    /// Embeds a watermark into the audio and returns the watermarked audio.
    /// </summary>
    public Vector<T> EmbedWatermark(Vector<T> audioSamples, int sampleRate)
        => EmbedWatermark(audioSamples, sampleRate, FrameBatch);

    /// <summary>
    /// The number of frames transformed per tensor batch, so the spectra of a long recording are never held at once.
    /// </summary>
    internal const int FrameBatch = 1024;

    /// <summary>
    /// <see cref="EmbedWatermark(Vector{T}, int)"/> with <paramref name="frameBatch"/> frames transformed at a time.
    /// Batches overlap-add into one output, and the chip sequence is indexed by the global frame, so the result does
    /// not depend on the batch size.
    /// </summary>
    internal Vector<T> EmbedWatermark(Vector<T> audioSamples, int sampleRate, int frameBatch)
    {
        if (frameBatch < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(frameBatch), "At least one frame must be processed per batch.");
        }

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

        var window = SafetyTensorOps<T>.SymmetricHann(frameSize);
        int halfLen = frameSize / 2;

        // Embed spread-spectrum watermark in mid-frequency band (1kHz - 6kHz)
        int lowBin = (int)(1000.0 * frameSize / sampleRate);
        int highBin = Math.Min((int)(6000.0 * frameSize / sampleRate), halfLen - 1);

        // Frames are transformed in bounded batches. Batch frames [first, first + count) cover samples
        // [first * hop, (first + count + 1) * hop); only the last hop of them is shared with the next batch, so the
        // rest is final and the shared hop is carried over.
        var resultParts = new List<Tensor<T>>();
        var weightParts = new List<Tensor<T>>();
        Tensor<T>? carry = null;
        Tensor<T>? carryWeight = null;
        for (int first = 0; first < numFrames; first += frameBatch)
        {
            int count = Math.Min(frameBatch, numFrames - first);
            int segmentLength = (count + 1) * hopSize;
            var segment = engine.TensorNarrow(audio, 0, first * hopSize, segmentLength);

            // Hann-windowed frames: [count, frameSize]
            var frames = engine.TensorMultiply(SafetyTensorOps<T>.Frames(segment, frameSize, hopSize, count), window);
            SafetyTensorOps<T>.Dft(frames, out var real, out var imag);

            if (highBin >= lowBin)
            {
                int bins = highBin - lowBin + 1;

                // Pseudo-random watermark chip based on key, bin and global frame index
                var chipValue = ChipValues(_watermarkKey, lowBin, bins, count, first);
                var bandReal = engine.TensorNarrow(real, 1, lowBin, bins);
                var bandImag = engine.TensorNarrow(imag, 1, lowBin, bins);
                var magnitude = engine.TensorSqrt(engine.TensorAdd(engine.TensorSquare(bandReal), engine.TensorSquare(bandImag)));
                var delta = engine.TensorMultiply(engine.TensorMultiplyScalar(magnitude, _watermarkStrength), chipValue);

                real = engine.TensorAdd(real, engine.TensorConcatenate(new[]
                {
                    Tensor<T>.CreateZeros(count, lowBin), delta,
                    Tensor<T>.CreateZeros(count, frameSize - lowBin - bins)
                }, 1));
            }

            engine.IFFT(real, imag, out var watermarkedFrames, out _);

            // Overlap-add within the batch
            var weighted = engine.TensorMultiply(watermarkedFrames, window);
            var windowEnergy = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorSquare(window), new[] { 1, frameSize }), new[] { count, frameSize });
            var (current, previous) = SafetyTensorOps<T>.HalfOverlapParts(weighted, segmentLength);
            var (currentWeight, previousWeight) = SafetyTensorOps<T>.HalfOverlapParts(windowEnergy, segmentLength);
            carry = AppendFinal(resultParts, engine.TensorAdd(previous, current), carry, hopSize);
            carryWeight = AppendFinal(weightParts, engine.TensorAdd(previousWeight, currentWeight), carryWeight, hopSize);
        }

        int coveredLength = (numFrames + 1) * hopSize;
        if (carry is not null && carryWeight is not null)
        {
            resultParts.Add(carry);
            weightParts.Add(carryWeight);
        }

        if (length > coveredLength)
        {
            resultParts.Add(Tensor<T>.CreateZeros(length - coveredLength));
            weightParts.Add(Tensor<T>.CreateZeros(length - coveredLength));
        }

        var result = engine.TensorConcatenate(resultParts.ToArray(), 0);
        var windowSum = engine.TensorConcatenate(weightParts.ToArray(), 0);

        // Normalize by window sum
        var covered = SafetyTensorOps<T>.Gt(windowSum, 1e-10);
        var normalized = engine.TensorDivide(result, SafetyTensorOps<T>.Where(covered, windowSum, 1.0));
        return SafetyTensorOps<T>.Where(covered, normalized, audio).ToVector();
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        // Detect our specific watermark
        var findings = new List<SafetyFinding>();

        if (audioSamples.Length < 512) return findings;

        T detectionScoreT = DetectWatermark(audioSamples, sampleRate);

        if (NumOps.GreaterThanOrEquals(detectionScoreT, NumOps.FromDouble(0.5)))
        {
            double detectionScore = NumOps.ToDouble(detectionScoreT);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = Math.Min(1.0, detectionScore),
                Description = $"Voice watermark detected (score: {detectionScore:F3}). " +
                              $"Audio contains embedded identification watermark.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private T DetectWatermark(Vector<T> audio, int sampleRate)
    {
        int frameSize = 512;
        int hopSize = 256;
        int numFrames = Math.Min(8, (audio.Length - frameSize) / hopSize + 1);
        if (numFrames < 1) return NumOps.Zero;

        int halfLen = frameSize / 2;
        int lowBin = (int)(1000.0 * frameSize / sampleRate);
        int highBin = Math.Min((int)(6000.0 * frameSize / sampleRate), halfLen - 1);
        if (highBin < lowBin) return NumOps.Zero;
        int bins = highBin - lowBin + 1;

        using var noGrad = new NoGradScope<T>();
        var engine = SafetyTensorOps<T>.Engine;
        var frames = engine.TensorMultiply(
            SafetyTensorOps<T>.Frames(SafetyTensorOps<T>.AsTensor(audio), frameSize, hopSize, numFrames),
            SafetyTensorOps<T>.SymmetricHann(frameSize));
        SafetyTensorOps<T>.Dft(frames, out var real, out _);

        // Correlate spectrum with expected watermark pattern
        var chipValue = ChipValues(_watermarkKey, lowBin, bins, numFrames);
        var corr = engine.ReduceSum(
            engine.TensorMultiply(engine.TensorNarrow(real, 1, lowBin, bins), chipValue), new[] { 1 }, false);
        T correlationSum = engine.TensorSum(engine.TensorDivideScalar(corr, NumOps.FromDouble(bins)));

        T avgCorrelation = NumOps.Divide(correlationSum, NumOps.FromDouble(numFrames));
        return SafetyTensorOps<T>.ClampUnit(NumOps.Multiply(avgCorrelation, NumOps.FromDouble(10)));
    }

    /// <summary>
    /// Adds the carried overlap to the start of a batch's overlap-added segment, appends every sample no later batch
    /// reaches to <paramref name="parts"/>, and returns the last <paramref name="hop"/> samples as the new carry.
    /// </summary>
    private static Tensor<T> AppendFinal(List<Tensor<T>> parts, Tensor<T> segment, Tensor<T>? carry, int hop)
    {
        var engine = SafetyTensorOps<T>.Engine;
        int length = segment.Shape[0];
        if (carry is not null)
        {
            segment = engine.TensorConcatenate(new[]
            {
                engine.TensorAdd(engine.TensorNarrow(segment, 0, 0, hop), carry),
                engine.TensorNarrow(segment, 0, hop, length - hop)
            }, 0);
        }

        parts.Add(engine.TensorNarrow(segment, 0, 0, length - hop).Contiguous());
        return engine.TensorNarrow(segment, 0, length - hop, hop).Contiguous();
    }

    /// <summary>
    /// The +1/-1 watermark chip of every (frame, bin) pair, <c>[frames, bins]</c> for bins
    /// <c>lowBin .. lowBin + bins - 1</c>: the low bit of the key/bin/frame hash
    /// (<c>key * 2654435761 + bin * 2246822519 + frame * 3266489917</c>, wrapping, then one
    /// finalizer round) maps 1 to +1 and 0 to -1.
    /// </summary>
    private static Tensor<T> ChipValues(int key, int lowBin, int bins, int frames, int firstFrame = 0)
    {
        var engine = SafetyTensorOps<T>.Engine;
        var grid = new[] { frames, bins };
        int keyTerm;
        unchecked
        {
            keyTerm = key * (int)2654435761;
        }

        var binTerm = engine.Reshape(
            engine.TensorMultiplyScalar(engine.TensorArange<int>(lowBin, lowBin + bins, 1), unchecked((int)2246822519)),
            new[] { 1, bins });
        var frameTerm = engine.Reshape(
            engine.TensorMultiplyScalar(engine.TensorArange<int>(firstFrame, firstFrame + frames, 1), unchecked((int)3266489917)),
            new[] { frames, 1 });
        var hash = engine.TensorAddScalar(
            engine.TensorAdd(engine.TensorBroadcastTo(binTerm, grid), engine.TensorBroadcastTo(frameTerm, grid)),
            keyTerm);
        var sixteen = Tensor<int>.CreateDefault(grid, 16);
        hash = engine.TensorMultiplyScalar(
            engine.TensorBitwiseXor(engine.TensorBitwiseRightShift(hash, sixteen), hash), 0x45d9f3b);
        hash = engine.TensorBitwiseXor(engine.TensorBitwiseRightShift(hash, sixteen), hash);
        var chip = engine.TensorTake(SafetyTensorOps<T>.Range(2),
            engine.TensorBitwiseAnd(hash, Tensor<int>.CreateDefault(grid, 1)));
        return SafetyTensorOps<T>.Affine(chip, 2.0, -1.0);
    }
}
