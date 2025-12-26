using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Adds background noise to audio data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This augmentation adds random noise to audio,
/// simulating real-world recording conditions like background hum, ambient sounds,
/// or electronic interference. This helps models become robust to noisy inputs.</para>
/// <para><b>SNR (Signal-to-Noise Ratio):</b>
/// <list type="bullet">
/// <item>Higher SNR = less noise (cleaner audio)</item>
/// <item>Lower SNR = more noise (noisier audio)</item>
/// <item>20 dB = barely audible noise</item>
/// <item>10 dB = noticeable noise</item>
/// <item>0 dB = signal and noise are equal</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AudioNoise<T> : AudioAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum signal-to-noise ratio in dB.
    /// </summary>
    public double MinSnrDb { get; }

    /// <summary>
    /// Gets the maximum signal-to-noise ratio in dB.
    /// </summary>
    public double MaxSnrDb { get; }

    /// <summary>
    /// Gets or sets the type of noise to add.
    /// </summary>
    public NoiseType NoiseType { get; set; } = NoiseType.White;

    /// <summary>
    /// Creates a new audio noise augmentation.
    /// </summary>
    /// <param name="minSnrDb">Minimum SNR in dB (default: 10.0).</param>
    /// <param name="maxSnrDb">Maximum SNR in dB (default: 30.0).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.3).</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz (default: 16000).</param>
    public AudioNoise(
        double minSnrDb = 10.0,
        double maxSnrDb = 30.0,
        double probability = 0.3,
        int sampleRate = 16000) : base(probability, sampleRate)
    {
        if (minSnrDb > maxSnrDb)
        {
            throw new ArgumentException("Minimum SNR must be less than or equal to maximum SNR.");
        }

        MinSnrDb = minSnrDb;
        MaxSnrDb = maxSnrDb;
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        double snrDb = context.GetRandomDouble(MinSnrDb, MaxSnrDb);
        return AddNoise(data, snrDb, context);
    }

    private Tensor<T> AddNoise(Tensor<T> waveform, double snrDb, AugmentationContext<T> context)
    {
        int samples = GetSampleCount(waveform);
        var result = waveform.Clone();

        // Calculate signal power
        double signalPower = 0;
        for (int i = 0; i < samples; i++)
        {
            double val = Convert.ToDouble(waveform[i]);
            signalPower += val * val;
        }
        signalPower /= samples;

        // Convert SNR from dB to linear ratio
        // SNR_db = 10 * log10(signal_power / noise_power)
        // noise_power = signal_power / 10^(SNR_db/10)
        double snrLinear = Math.Pow(10, snrDb / 10.0);
        double noisePower = signalPower / snrLinear;
        double noiseStd = Math.Sqrt(noisePower);

        // Add noise
        for (int i = 0; i < samples; i++)
        {
            double noise = GenerateNoise(context, noiseStd);
            double originalValue = Convert.ToDouble(waveform[i]);
            double newValue = originalValue + noise;

            // Clip to valid range [-1, 1]
            newValue = Math.Max(-1.0, Math.Min(1.0, newValue));
            result[i] = (T)Convert.ChangeType(newValue, typeof(T));
        }

        return result;
    }

    private double GenerateNoise(AugmentationContext<T> context, double stdDev)
    {
        return NoiseType switch
        {
            NoiseType.White => context.SampleGaussian(0, stdDev),
            NoiseType.Pink => GeneratePinkNoise(context, stdDev),
            NoiseType.Brown => GenerateBrownNoise(context, stdDev),
            _ => context.SampleGaussian(0, stdDev)
        };
    }

    // Simplified pink noise using Voss-McCartney algorithm approximation
    private double _pinkNoiseState = 0;
    private double GeneratePinkNoise(AugmentationContext<T> context, double stdDev)
    {
        // Pink noise has 1/f spectrum - simplified implementation
        double white = context.SampleGaussian(0, stdDev);
        _pinkNoiseState = 0.997 * _pinkNoiseState + 0.029 * white;
        return _pinkNoiseState + white * 0.1;
    }

    // Brown noise (random walk)
    private double _brownNoiseState = 0;
    private double GenerateBrownNoise(AugmentationContext<T> context, double stdDev)
    {
        double white = context.SampleGaussian(0, stdDev);
        _brownNoiseState += white * 0.02;
        // Leaky integrator to prevent drift
        _brownNoiseState *= 0.999;
        return _brownNoiseState;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minSnrDb"] = MinSnrDb;
        parameters["maxSnrDb"] = MaxSnrDb;
        parameters["noiseType"] = NoiseType.ToString();
        return parameters;
    }
}

/// <summary>
/// Types of audio noise.
/// </summary>
public enum NoiseType
{
    /// <summary>
    /// White noise (flat spectrum, all frequencies equal).
    /// </summary>
    White,

    /// <summary>
    /// Pink noise (1/f spectrum, more natural sounding).
    /// </summary>
    Pink,

    /// <summary>
    /// Brown/Brownian noise (1/f² spectrum, deep rumble).
    /// </summary>
    Brown
}
