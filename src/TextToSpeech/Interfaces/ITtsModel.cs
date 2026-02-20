namespace AiDotNet.TextToSpeech.Interfaces;

/// <summary>
/// Base interface for all text-to-speech models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TTS models convert text input into audio waveform output. This base interface defines
/// the core synthesis method shared by all TTS architectures:
/// <list type="bullet">
/// <item>Acoustic models (Tacotron, FastSpeech) that generate mel-spectrograms</item>
/// <item>Vocoders (HiFi-GAN, WaveNet) that convert mel-spectrograms to waveforms</item>
/// <item>End-to-end models (VITS) that go directly from text to waveform</item>
/// <item>Codec-based models (VALL-E, CosyVoice) that use neural audio codecs</item>
/// </list>
/// </para>
/// </remarks>
public interface ITtsModel<T>
{
    /// <summary>
    /// Synthesizes audio waveform from text input.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>Audio waveform tensor.</returns>
    Tensor<T> Synthesize(string text);

    /// <summary>
    /// Gets the audio sample rate in Hz (e.g., 22050, 24000, 44100).
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the maximum input text length in characters.
    /// </summary>
    int MaxTextLength { get; }
}
