namespace AiDotNet.TextToSpeech.Interfaces;

/// <summary>
/// Interface for codec-based TTS models that use neural audio codecs with language model decoding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Codec-based TTS models use a neural audio codec (e.g., EnCodec, SoundStream, DAC) to represent
/// audio as discrete tokens, then use a language model to predict those tokens from text:
/// Text -> [LM] -> Audio Tokens -> [Codec Decoder] -> Waveform.
/// Architectures include:
/// <list type="bullet">
/// <item>AR + NAR: VALL-E (autoregressive first codebook + non-autoregressive rest)</item>
/// <item>Flow matching: CosyVoice, Voicebox (conditional flow matching on codec tokens)</item>
/// <item>LLM-based: Fish Speech, Llasa (fine-tuned LLM predicts codec tokens)</item>
/// <item>Parallel: SoundStorm (MaskGIT-style parallel decoding)</item>
/// </list>
/// </para>
/// </remarks>
public interface ICodecTts<T> : ITtsModel<T>
{
    /// <summary>
    /// Encodes audio into discrete codec tokens.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Codec token tensor.</returns>
    Tensor<T> EncodeToTokens(Tensor<T> audio);

    /// <summary>
    /// Decodes codec tokens back to audio waveform.
    /// </summary>
    /// <param name="tokens">Codec token tensor.</param>
    /// <returns>Audio waveform tensor.</returns>
    Tensor<T> DecodeFromTokens(Tensor<T> tokens);

    /// <summary>
    /// Gets the number of residual vector quantization codebooks.
    /// </summary>
    int NumCodebooks { get; }

    /// <summary>
    /// Gets the codebook vocabulary size.
    /// </summary>
    int CodebookSize { get; }

    /// <summary>
    /// Gets the codec frame rate in Hz (tokens per second of audio).
    /// </summary>
    int CodecFrameRate { get; }
}
