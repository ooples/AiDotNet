namespace AiDotNet.TextToSpeech.Interfaces;

/// <summary>
/// Interface for neural vocoders that convert mel-spectrograms to audio waveforms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Vocoders form the second stage of a two-stage TTS pipeline:
/// Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.
/// Architectures include:
/// <list type="bullet">
/// <item>Autoregressive: WaveNet, WaveRNN (sample-by-sample generation)</item>
/// <item>GAN-based: HiFi-GAN, MelGAN, BigVGAN (adversarial training, parallel)</item>
/// <item>Flow-based: WaveGlow (invertible flow, parallel)</item>
/// <item>Diffusion-based: DiffWave, WaveGrad (denoising diffusion)</item>
/// <item>Fourier-based: Vocos, iSTFTNet (inverse STFT output)</item>
/// </list>
/// </para>
/// </remarks>
public interface IVocoder<T>
{
    /// <summary>
    /// Converts a mel-spectrogram to an audio waveform.
    /// </summary>
    /// <param name="melSpectrogram">Mel-spectrogram tensor of shape [mel_channels, time_steps].</param>
    /// <returns>Audio waveform tensor of shape [num_samples].</returns>
    Tensor<T> MelToWaveform(Tensor<T> melSpectrogram);

    /// <summary>
    /// Gets the audio sample rate in Hz (e.g., 22050, 24000).
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the expected number of mel channels in the input spectrogram.
    /// </summary>
    int MelChannels { get; }

    /// <summary>
    /// Gets the upsampling factor (hop_size) from mel frames to audio samples.
    /// </summary>
    int UpsampleFactor { get; }
}
