using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Stretches or compresses audio in time without changing pitch.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Time stretching makes audio faster or slower without
/// changing the pitch - like how a slower speaker still has the same voice pitch.
/// This is different from simply playing at a different speed.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Speech recognition to handle different speaking speeds</item>
/// <item>Music tempo adjustment</item>
/// <item>Synchronizing audio with video</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TimeStretch<T> : AudioAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum time stretch factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.8 (20% faster)</para>
    /// <para>Values less than 1.0 speed up the audio.</para>
    /// </remarks>
    public double MinRate { get; }

    /// <summary>
    /// Gets the maximum time stretch factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: 1.2 (20% slower)</para>
    /// <para>Values greater than 1.0 slow down the audio.</para>
    /// </remarks>
    public double MaxRate { get; }

    /// <summary>
    /// Creates a new time stretch augmentation.
    /// </summary>
    /// <param name="minRate">Minimum stretch rate (default: 0.8).</param>
    /// <param name="maxRate">Maximum stretch rate (default: 1.2).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz (default: 16000).</param>
    public TimeStretch(
        double minRate = 0.8,
        double maxRate = 1.2,
        double probability = 0.5,
        int sampleRate = 16000) : base(probability, sampleRate)
    {
        if (minRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minRate), "Minimum rate must be positive.");
        }

        if (maxRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxRate), "Maximum rate must be positive.");
        }

        if (minRate > maxRate)
        {
            throw new ArgumentException("Minimum rate must be less than or equal to maximum rate.");
        }

        MinRate = minRate;
        MaxRate = maxRate;
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        double rate = context.GetRandomDouble(MinRate, MaxRate);
        return ApplyTimeStretch(data, rate);
    }

    private Tensor<T> ApplyTimeStretch(Tensor<T> waveform, double rate)
    {
        int originalSamples = GetSampleCount(waveform);
        int newLength = (int)(originalSamples * rate);
        if (newLength < 1) newLength = 1;

        // Create new shape with modified time dimension
        var newShape = (int[])waveform._shape.Clone();
        newShape[waveform.Rank - 1] = newLength;
        var result = new Tensor<T>(newShape);

        // Calculate total elements excluding time dimension (handles multi-channel audio)
        int outerElements = 1;
        for (int d = 0; d < waveform.Rank - 1; d++)
        {
            outerElements *= waveform.Shape[d];
        }

        // Process each channel/batch independently using linear interpolation
        // Production implementation would use phase vocoder for better quality
        var engine = Engine;
        var rows = engine.Reshape(waveform, [outerElements, originalSamples]);

        // Source position of every output sample: i * originalSamples / newLength. Positions and the
        // interpolation are evaluated in double, so an integral T cannot truncate the fractional weights
        var srcPos = engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<double>(0, newLength, 1), (double)originalSamples),
            (double)newLength);
        var srcFloor = engine.TensorFloor(srcPos);
        double last = originalSamples - 1;

        // Positions at or past the last sample take the last sample as is (index clamped, weight 0)
        var atEnd = engine.TensorGreaterThan(srcFloor, originalSamples - 1.5);
        var lower = engine.TensorClampMax(srcFloor, last).Cast<int>();
        var upper = engine.TensorClampMax(engine.TensorAddScalar(srcFloor, 1.0), last).Cast<int>();
        var frac = engine.TensorMultiply(engine.TensorSubtract(srcPos, srcFloor), engine.ScalarMinusTensor(1.0, atEnd));

        int[] outShape = [outerElements, newLength];
        var wide = rows.Contiguous().Cast<double>();
        var val1 = engine.TensorIndexSelect(wide, lower, 1);
        var val2 = engine.TensorIndexSelect(wide, upper, 1);
        var weight = engine.TensorBroadcastTo(engine.Reshape(frac, [1, newLength]), outShape);
        var blended = engine.TensorAdd(val1, engine.TensorMultiply(weight, engine.TensorSubtract(val2, val1))).Cast<T>();
        var endMask = engine.TensorBroadcastTo(engine.Reshape(atEnd.Cast<T>(), [1, newLength]), outShape);
        var interpolated = engine.TensorWhere(endMask, engine.TensorIndexSelect(rows, lower, 1), blended);
        interpolated.Contiguous().CopyTo(result.AsWritableSpan());

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minRate"] = MinRate;
        parameters["maxRate"] = MaxRate;
        return parameters;
    }
}
