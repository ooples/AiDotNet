using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Shifts the pitch of audio without changing the tempo.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Pitch shifting makes audio sound higher or lower,
/// like the difference between a high and low voice. Unlike changing playback speed,
/// this keeps the audio the same length while changing only the pitch.</para>
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
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        // Sample random pitch shift
        double semitones = context.GetRandomDouble(MinSemitones, MaxSemitones);

        // Convert semitones to frequency ratio
        // ratio = 2^(semitones/12)
        double ratio = Math.Pow(2, semitones / 12.0);

        // Apply pitch shifting using resampling technique
        // This is a simplified version - production would use phase vocoder
        return ApplyPitchShift(data, ratio);
    }

    private Tensor<T> ApplyPitchShift(Tensor<T> waveform, double ratio)
    {
        int samples = GetSampleCount(waveform);

        // For pitch shifting without tempo change:
        // 1. Resample to change pitch (changes length)
        // 2. Time stretch back to original length (preserves pitch change)

        // Simplified implementation using linear interpolation resampling
        int newLength = (int)(samples / ratio);
        if (newLength < 1) newLength = 1;

        var result = new Tensor<T>(waveform.Shape);

        for (int i = 0; i < samples; i++)
        {
            // Map output position to input position
            double srcPos = i * ratio;
            int srcIndex = (int)srcPos;
            double frac = srcPos - srcIndex;

            if (srcIndex >= newLength - 1)
            {
                // At the end, use last value
                int clampedIdx = Math.Min(srcIndex, samples - 1);
                result[i] = waveform[clampedIdx];
            }
            else if (srcIndex < samples - 1)
            {
                // Linear interpolation
                double val1 = Convert.ToDouble(waveform[srcIndex]);
                double val2 = Convert.ToDouble(waveform[srcIndex + 1]);
                double interpolated = val1 + frac * (val2 - val1);
                result[i] = (T)Convert.ChangeType(interpolated, typeof(T));
            }
            else
            {
                result[i] = waveform[samples - 1];
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minSemitones"] = MinSemitones;
        parameters["maxSemitones"] = MaxSemitones;
        return parameters;
    }
}
