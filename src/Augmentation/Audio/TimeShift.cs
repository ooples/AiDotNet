using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Shifts audio forward or backward in time.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Time shifting moves audio forward or backward,
/// like adding silence at the beginning or end. This simulates different
/// recording start times and helps models handle timing variations.</para>
/// <para><b>Handling shifted samples:</b>
/// <list type="bullet">
/// <item>Wrap: Samples that go off one end appear at the other (circular)</item>
/// <item>Zero: Shifted areas are filled with silence</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TimeShift<T> : AudioAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum shift as a fraction of total duration.
    /// </summary>
    /// <remarks>
    /// <para>Default: -0.2 (20% backward)</para>
    /// <para>Negative values shift audio backward in time (content moves earlier, silence fills end).</para>
    /// </remarks>
    public double MinShiftFraction { get; }

    /// <summary>
    /// Gets the maximum shift as a fraction of total duration.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.2 (20% forward)</para>
    /// <para>Positive values shift audio forward in time (content moves later, silence fills start).</para>
    /// </remarks>
    public double MaxShiftFraction { get; }

    /// <summary>
    /// Gets or sets whether to wrap shifted samples (true) or fill with zeros (false).
    /// </summary>
    public bool WrapAround { get; set; } = false;

    /// <summary>
    /// Creates a new time shift augmentation.
    /// </summary>
    /// <param name="minShiftFraction">Minimum shift fraction (default: -0.2).</param>
    /// <param name="maxShiftFraction">Maximum shift fraction (default: 0.2).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz (default: 16000).</param>
    public TimeShift(
        double minShiftFraction = -0.2,
        double maxShiftFraction = 0.2,
        double probability = 0.5,
        int sampleRate = 16000) : base(probability, sampleRate)
    {
        if (minShiftFraction > maxShiftFraction)
        {
            throw new ArgumentException("Minimum shift must be less than or equal to maximum shift.");
        }

        if (minShiftFraction < -1.0 || maxShiftFraction > 1.0)
        {
            throw new ArgumentOutOfRangeException(
                "Shift fractions must be between -1.0 and 1.0.");
        }

        MinShiftFraction = minShiftFraction;
        MaxShiftFraction = maxShiftFraction;
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        double shiftFraction = context.GetRandomDouble(MinShiftFraction, MaxShiftFraction);
        return ApplyTimeShift(data, shiftFraction);
    }

    private Tensor<T> ApplyTimeShift(Tensor<T> waveform, double shiftFraction)
    {
        int samples = GetSampleCount(waveform);
        int shiftSamples = (int)(samples * shiftFraction);

        if (shiftSamples == 0)
        {
            return waveform.Clone();
        }

        var result = new Tensor<T>(waveform.Shape);
        T zero = NumOps.Zero;

        for (int i = 0; i < samples; i++)
        {
            int srcIndex = i - shiftSamples;

            if (WrapAround)
            {
                // Wrap around: samples that go off one end appear at the other
                srcIndex = ((srcIndex % samples) + samples) % samples;
                result[i] = waveform[srcIndex];
            }
            else
            {
                // Fill with zeros: out-of-bounds areas are silent
                if (srcIndex >= 0 && srcIndex < samples)
                {
                    result[i] = waveform[srcIndex];
                }
                else
                {
                    result[i] = zero;
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minShiftFraction"] = MinShiftFraction;
        parameters["maxShiftFraction"] = MaxShiftFraction;
        parameters["wrapAround"] = WrapAround;
        return parameters;
    }
}
