using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Randomly changes the volume (gain) of audio.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Volume change augmentation makes audio louder or quieter.
/// This helps models become robust to different recording volumes and microphone
/// distances.</para>
/// <para><b>Gain in dB:</b>
/// <list type="bullet">
/// <item>+6 dB ≈ 2x louder</item>
/// <item>0 dB = no change</item>
/// <item>-6 dB ≈ 0.5x volume</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VolumeChange<T> : AudioAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum volume change in dB.
    /// </summary>
    public double MinGainDb { get; }

    /// <summary>
    /// Gets the maximum volume change in dB.
    /// </summary>
    public double MaxGainDb { get; }

    /// <summary>
    /// Creates a new volume change augmentation.
    /// </summary>
    /// <param name="minGainDb">Minimum gain in dB (default: -6.0).</param>
    /// <param name="maxGainDb">Maximum gain in dB (default: 6.0).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz (default: 16000).</param>
    public VolumeChange(
        double minGainDb = -6.0,
        double maxGainDb = 6.0,
        double probability = 0.5,
        int sampleRate = 16000) : base(probability, sampleRate)
    {
        if (minGainDb > maxGainDb)
        {
            throw new ArgumentException("Minimum gain must be less than or equal to maximum gain.");
        }

        MinGainDb = minGainDb;
        MaxGainDb = maxGainDb;
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        double gainDb = context.GetRandomDouble(MinGainDb, MaxGainDb);
        return ApplyGain(data, gainDb);
    }

    private Tensor<T> ApplyGain(Tensor<T> waveform, double gainDb)
    {
        // Convert dB to linear scale: gain = 10^(dB/20)
        double gainLinear = Math.Pow(10, gainDb / 20.0);

        int samples = GetSampleCount(waveform);
        var result = waveform.Clone();

        for (int i = 0; i < samples; i++)
        {
            double originalValue = NumOps.ToDouble(waveform[i]);
            double newValue = originalValue * gainLinear;

            // Clip to valid range [-1, 1] to prevent clipping
            newValue = Math.Max(-1.0, Math.Min(1.0, newValue));
            result[i] = NumOps.FromDouble(newValue);
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minGainDb"] = MinGainDb;
        parameters["maxGainDb"] = MaxGainDb;
        return parameters;
    }
}
