using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Base class for audio data augmentations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Audio augmentation transforms sound data to improve model
/// robustness to variations in recording conditions, speaking styles, and environmental noise.
/// Common techniques include:
/// <list type="bullet">
/// <item>Time stretching (faster/slower without pitch change)</item>
/// <item>Pitch shifting (higher/lower without speed change)</item>
/// <item>Adding background noise</item>
/// <item>Volume changes</item>
/// <item>Time shifting (moving audio forward/backward)</item>
/// </list>
/// </para>
/// <para>Audio data is typically represented as a 1D waveform tensor or 2D spectrogram.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class AudioAugmenterBase<T> : AugmentationBase<T, Tensor<T>>
{
    /// <summary>
    /// Gets or sets the sample rate of the audio data in Hz.
    /// </summary>
    /// <remarks>
    /// <para>Default: 16000 Hz (common for speech recognition)</para>
    /// <para>Other common values: 22050 Hz (music), 44100 Hz (CD quality), 48000 Hz (professional audio)</para>
    /// </remarks>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Initializes a new audio augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation (0.0 to 1.0).</param>
    /// <param name="sampleRate">The sample rate of the audio data in Hz.</param>
    protected AudioAugmenterBase(double probability = 1.0, int sampleRate = 16000) : base(probability)
    {
        SampleRate = sampleRate;
    }

    /// <summary>
    /// Gets the duration of the audio in seconds.
    /// </summary>
    /// <param name="waveform">The audio waveform tensor.</param>
    /// <returns>The duration in seconds.</returns>
    protected double GetDuration(Tensor<T> waveform)
    {
        // Assume the last dimension is time
        var samples = waveform.Shape[waveform.Rank - 1];
        return (double)samples / SampleRate;
    }

    /// <summary>
    /// Gets the number of audio samples.
    /// </summary>
    /// <param name="waveform">The audio waveform tensor.</param>
    /// <returns>The number of samples.</returns>
    protected int GetSampleCount(Tensor<T> waveform)
    {
        return waveform.Shape[waveform.Rank - 1];
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["sampleRate"] = SampleRate;
        return parameters;
    }
}
