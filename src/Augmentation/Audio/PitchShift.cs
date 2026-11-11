using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Shifts the pitch of audio without changing tempo using WSOLA (Waveform Similarity Overlap-Add).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Pitch shifting makes audio sound higher or lower,
/// like the difference between a high and low voice, while keeping the same duration.</para>
/// <para><b>Algorithm:</b> Uses WSOLA (Waveform Similarity Overlap-Add) for time-stretching
/// followed by resampling. This preserves audio quality better than simple resampling.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Speech recognition to handle different voice pitches</item>
/// <item>Music analysis to handle different keys</item>
/// <item>Voice cloning and synthesis training</item>
/// </list>
/// </para>
/// <para><b>Semitone reference:</b> 12 semitones = 1 octave (doubling/halving frequency)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PitchShift<T> : AudioAugmenterBase<T>
{
    // WSOLA parameters
    private readonly int _frameSize;
    private readonly int _hopSize;
    private readonly int _searchRange;

    /// <summary>
    /// Gets the minimum pitch shift in semitones.
    /// </summary>
    /// <remarks>
    /// <para>Default: -2.0 semitones (about -12% frequency)</para>
    /// <para>Negative values lower the pitch.</para>
    /// </remarks>
    public double MinSemitones { get; }

    /// <summary>
    /// Gets the maximum pitch shift in semitones.
    /// </summary>
    /// <remarks>
    /// <para>Default: 2.0 semitones (about +12% frequency)</para>
    /// <para>Positive values raise the pitch.</para>
    /// </remarks>
    public double MaxSemitones { get; }

    /// <summary>
    /// Creates a new pitch shift augmentation.
    /// </summary>
    /// <param name="minSemitones">Minimum pitch shift in semitones (default: -2.0).</param>
    /// <param name="maxSemitones">Maximum pitch shift in semitones (default: 2.0).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz (default: 16000).</param>
    public PitchShift(
        double minSemitones = -2.0,
        double maxSemitones = 2.0,
        double probability = 0.5,
        int sampleRate = 16000) : base(probability, sampleRate)
    {
        if (minSemitones > maxSemitones)
        {
            throw new ArgumentException("Minimum semitones must be less than or equal to maximum semitones.");
        }

        MinSemitones = minSemitones;
        MaxSemitones = maxSemitones;

        // WSOLA frame parameters based on sample rate
        // Frame size ~25ms, hop size ~6.25ms (75% overlap)
        _frameSize = Math.Max(256, (int)(sampleRate * 0.025));
        _hopSize = _frameSize / 4;
        _searchRange = _hopSize / 2;
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        // Sample random pitch shift
        double semitones = context.GetRandomDouble(MinSemitones, MaxSemitones);

        // Convert semitones to frequency ratio
        // ratio = 2^(semitones/12)
        double pitchRatio = Math.Pow(2, semitones / 12.0);

        if (Math.Abs(pitchRatio - 1.0) < 0.001)
        {
            return data.Clone();
        }

        // Apply pitch shifting: time-stretch then resample
        return ApplyPitchShift(data, pitchRatio);
    }

    /// <summary>
    /// Applies pitch shift using WSOLA time-stretching followed by resampling.
    /// </summary>
    private Tensor<T> ApplyPitchShift(Tensor<T> waveform, double pitchRatio)
    {
        int samples = GetSampleCount(waveform);
        if (samples < _frameSize)
        {
            return waveform.Clone();
        }

        // Work on the first `samples` values as a flat tensor
        var engine = Engine;
        var input = engine.TensorSlice(engine.Reshape(waveform, [waveform.Length]), [0], [samples]).Contiguous();

        // Step 1: Time-stretch by pitchRatio using WSOLA
        // To shift pitch up by ratio, we stretch time by ratio, then resample back
        var stretched = TimeStretchWSOLA(engine, input, pitchRatio);

        // Step 2: Resample back to original length
        var output = ResampleLinear(engine, stretched, samples);

        // Copy into a tensor of the original shape (values past `samples` stay zero)
        var result = new Tensor<T>(waveform._shape);
        output.Contiguous().CopyTo(result.AsWritableSpan().Slice(0, samples));

        return result;
    }

    /// <summary>
    /// Time-stretches audio using WSOLA algorithm.
    /// </summary>
    /// <param name="engine">The engine used for the tensor operations.</param>
    /// <param name="input">Input audio samples.</param>
    /// <param name="stretchFactor">Factor to stretch by (>1 = longer, &lt;1 = shorter).</param>
    /// <returns>Time-stretched audio.</returns>
    private Tensor<T> TimeStretchWSOLA(IEngine engine, Tensor<T> input, double stretchFactor)
    {
        int inputLength = input.Length;
        int outputLength = (int)(inputLength * stretchFactor);
        if (outputLength < 1) outputLength = 1;

        var output = new Tensor<T>([outputLength]);
        var windowWeights = new Tensor<T>([outputLength]);

        // Precompute Hann window: 0.5 * (1 - cos(2 pi i / (frameSize - 1)))
        var phase = engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(0, _frameSize, 1), NumOps.FromDouble(2 * Math.PI)),
            NumOps.FromDouble(_frameSize - 1));
        var hannWindow = engine.TensorMultiplyScalar(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorCos(phase)), NumOps.FromDouble(0.5));

        // Synthesis hop size (input hop scaled by stretch factor)
        int synthesisHop = Math.Max(1, (int)(_hopSize * stretchFactor));

        // Analysis position tracks where we read from input
        double analysisPos = 0;
        int synthesisPos = 0;

        while (synthesisPos + _frameSize <= outputLength && (int)analysisPos + _frameSize <= inputLength)
        {
            int currentAnalysisPos = (int)analysisPos;

            // For subsequent frames, find best match using cross-correlation
            if (synthesisPos > 0)
            {
                int bestOffset = FindBestOffset(engine, input, output, currentAnalysisPos, synthesisPos, inputLength);
                currentAnalysisPos = Math.Max(0, Math.Min(inputLength - _frameSize, currentAnalysisPos + bestOffset));
            }

            // Overlap-add the frame with Hann window (the loop bounds keep the whole frame in range)
            var frame = engine.TensorMultiply(engine.TensorSlice(input, [currentAnalysisPos], [_frameSize]), hannWindow);
            // Only the frame's span is written back, so each frame costs O(frameSize), not O(output)
            engine.TensorAdd(engine.TensorSlice(output, [synthesisPos], [_frameSize]), frame).Contiguous()
                .CopyTo(output.AsWritableSpan().Slice(synthesisPos, _frameSize));
            engine.TensorAdd(engine.TensorSlice(windowWeights, [synthesisPos], [_frameSize]), hannWindow).Contiguous()
                .CopyTo(windowWeights.AsWritableSpan().Slice(synthesisPos, _frameSize));

            analysisPos += _hopSize;
            synthesisPos += synthesisHop;
        }

        // Normalize by window weights to prevent amplitude changes
        var covered = engine.TensorGreaterThan(windowWeights, NumOps.FromDouble(0.001));
        var safeWeights = engine.TensorWhere(covered, windowWeights, OnesLike(engine, windowWeights));
        return engine.TensorWhere(covered, engine.TensorDivide(output, safeWeights), output);
    }

    /// <summary>
    /// Offset in [-searchRange, searchRange] (kept inside the input) whose input window best matches,
    /// by normalized cross-correlation, the last frameSize / 4 output samples before the synthesis
    /// position; the first best offset wins, and 0 is returned when the overlap is under 4 samples.
    /// </summary>
    private int FindBestOffset(IEngine engine, Tensor<T> input, Tensor<T> output, int analysisPos, int synthesisPos, int inputLength)
    {
        // Search for best matching position within search range
        int searchStart = -Math.Min(_searchRange, analysisPos);
        int searchEnd = Math.Min(_searchRange, inputLength - analysisPos - _frameSize);

        // Use overlap region for correlation (last part of previous frame)
        int overlapSize = Math.Min(_frameSize / 4, synthesisPos);
        if (overlapSize < 4 || searchEnd < searchStart)
        {
            return 0;
        }

        // Every candidate window at once: [offsets, overlapSize]
        int offsets = searchEnd - searchStart + 1;
        var region = engine.TensorSlice(input, [analysisPos + searchStart], [offsets + overlapSize - 1]);
        var windows = engine.TensorUnfold(region, 0, overlapSize, 1).Contiguous();
        var reference = engine.TensorSlice(output, [synthesisPos - overlapSize], [overlapSize]).Contiguous();

        // Compute normalized cross-correlation (raw correlation when the norm product is tiny)
        var correlation = engine.TensorMatMul(windows, engine.Reshape(reference, [overlapSize, 1]));
        T referenceNorm = engine.TensorSum(engine.TensorMultiply(reference, reference));
        var windowNorms = engine.ReduceSum(engine.TensorMultiply(windows, windows), [1], true);
        var denominator = engine.TensorSqrt(engine.TensorMultiplyScalar(windowNorms, referenceNorm));
        var usable = engine.TensorGreaterThan(denominator, NumOps.FromDouble(0.0001));
        var safeDenominator = engine.TensorWhere(usable, denominator, OnesLike(engine, denominator));
        var score = engine.TensorWhere(usable, engine.TensorDivide(correlation, safeDenominator), correlation);

        int best = engine.TensorArgMax(engine.Reshape(score, [offsets]), 0).GetFlat(0);
        return searchStart + best;
    }

    /// <summary>
    /// Linear resampling to <paramref name="targetLength"/> samples: output i reads source position
    /// i * (inputLength - 1) / (targetLength - 1); positions at or past the last sample take it.
    /// </summary>
    /// <summary>
    /// Returns a tensor shaped like <paramref name="like"/> with every element set to one.
    /// </summary>
    private static Tensor<T> OnesLike(IEngine engine, Tensor<T> like)
    {
        var ones = new Tensor<T>((int[])like._shape.Clone());
        engine.TensorFill(ones, NumOps.One);
        return ones;
    }

    private static Tensor<T> ResampleLinear(IEngine engine, Tensor<T> input, int targetLength)
    {
        int inputLength = input.Length;
        T ratio = NumOps.FromDouble((double)(inputLength - 1) / (targetLength - 1));
        T last = NumOps.FromDouble(inputLength - 1);

        var srcPos = engine.TensorMultiplyScalar(engine.TensorArange<T>(0, targetLength, 1), ratio);
        var srcIndex = engine.TensorFloor(srcPos);
        var atEnd = engine.TensorGreaterThan(srcIndex, NumOps.FromDouble(inputLength - 1.5));
        var lower = engine.TensorClampMax(srcIndex, last);
        var upper = engine.TensorClampMax(engine.TensorAddScalar(srcIndex, NumOps.One), last);
        var frac = engine.TensorMultiply(engine.TensorSubtract(srcPos, srcIndex), engine.ScalarMinusTensor(NumOps.One, atEnd));

        var value0 = engine.TensorTake(input, lower.Cast<int>());
        var value1 = engine.TensorTake(input, upper.Cast<int>());
        return engine.TensorAdd(
            engine.TensorMultiply(value0, engine.ScalarMinusTensor(NumOps.One, frac)),
            engine.TensorMultiply(value1, frac));
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minSemitones"] = MinSemitones;
        parameters["maxSemitones"] = MaxSemitones;
        parameters["algorithm"] = "WSOLA";
        return parameters;
    }
}
