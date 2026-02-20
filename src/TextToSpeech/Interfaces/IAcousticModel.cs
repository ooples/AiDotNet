namespace AiDotNet.TextToSpeech.Interfaces;

/// <summary>
/// Interface for acoustic models that generate mel-spectrograms from text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Acoustic models form the first stage of a two-stage TTS pipeline:
/// Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.
/// Architectures include:
/// <list type="bullet">
/// <item>Autoregressive: Tacotron, Tacotron 2 (attention-based seq2seq)</item>
/// <item>Non-autoregressive: FastSpeech, FastSpeech 2 (parallel with duration predictor)</item>
/// <item>Flow-based: Glow-TTS (monotonic alignment search + flow)</item>
/// <item>Diffusion-based: Grad-TTS (score-based diffusion decoder)</item>
/// </list>
/// </para>
/// </remarks>
public interface IAcousticModel<T> : ITtsModel<T>
{
    /// <summary>
    /// Generates a mel-spectrogram from text input.
    /// </summary>
    /// <param name="text">The input text.</param>
    /// <returns>Mel-spectrogram tensor of shape [mel_channels, time_steps].</returns>
    Tensor<T> TextToMel(string text);

    /// <summary>
    /// Gets the number of mel-spectrogram frequency channels (typically 80).
    /// </summary>
    int MelChannels { get; }

    /// <summary>
    /// Gets the mel-spectrogram hop size in audio samples.
    /// </summary>
    int HopSize { get; }

    /// <summary>
    /// Gets the FFT window size used for spectrogram computation.
    /// </summary>
    int FftSize { get; }
}
