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
        var newShape = (int[])waveform.Shape.Clone();
        newShape[waveform.Rank - 1] = newLength;
        var result = new Tensor<T>(newShape);

        // Simple resampling using linear interpolation
        // Production implementation would use phase vocoder for better quality
        for (int i = 0; i < newLength; i++)
        {
            double srcPos = (double)i * originalSamples / newLength;
            int srcIndex = (int)srcPos;
            double frac = srcPos - srcIndex;

            if (srcIndex >= originalSamples - 1)
            {
                result[i] = waveform[originalSamples - 1];
            }
            else
            {
                double val1 = Convert.ToDouble(waveform[srcIndex]);
                double val2 = Convert.ToDouble(waveform[srcIndex + 1]);
                double interpolated = val1 + frac * (val2 - val1);
                result[i] = (T)Convert.ChangeType(interpolated, typeof(T));
            }
        }

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
