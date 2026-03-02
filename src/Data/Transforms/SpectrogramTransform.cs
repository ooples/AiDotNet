using AiDotNet.Diffusion.Audio;

namespace AiDotNet.Data.Transforms;

/// <summary>
/// Transforms raw audio waveform tensors into Mel spectrogram representations.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Wraps the existing <see cref="MelSpectrogram{T}"/> to implement <see cref="ITransform{TInput,TOutput}"/>,
/// enabling composable use in data pipelines.
/// </para>
/// <para><b>For Beginners:</b> A spectrogram is a visual representation of audio frequencies over time.
/// Converting raw audio to a spectrogram is a common preprocessing step for audio ML models.
/// <code>
/// var transform = new SpectrogramTransform&lt;float&gt;(sampleRate: 16000, nMels: 80);
/// Tensor&lt;float&gt; melSpec = transform.Apply(rawAudioTensor);
/// </code>
/// </para>
/// </remarks>
public class SpectrogramTransform<T> : ITransform<Tensor<T>, Tensor<T>>
{
    private readonly MelSpectrogram<T> _melSpec;

    /// <summary>
    /// Creates a new spectrogram transform.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000.</param>
    /// <param name="nMels">Number of Mel frequency bins. Default is 80.</param>
    /// <param name="nFft">FFT window size. Default is 400.</param>
    /// <param name="hopLength">Number of samples between STFT columns. Default is 160.</param>
    /// <param name="logMel">Whether to apply log compression. Default is true.</param>
    public SpectrogramTransform(
        int sampleRate = 16000,
        int nMels = 80,
        int nFft = 400,
        int hopLength = 160,
        bool logMel = true)
    {
        _melSpec = new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: nMels,
            nFft: nFft,
            hopLength: hopLength,
            logMel: logMel);
    }

    /// <summary>
    /// Applies the Mel spectrogram transform to a raw audio waveform tensor.
    /// </summary>
    /// <param name="input">Raw audio waveform tensor of shape [samples] or [batch, samples].</param>
    /// <returns>Mel spectrogram tensor of shape [frames, nMels] or [batch, frames, nMels].</returns>
    public Tensor<T> Apply(Tensor<T> input)
    {
        return _melSpec.Forward(input);
    }
}
