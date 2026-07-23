namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for neural audio codecs that compress and decompress audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Neural audio codecs use neural networks to compress audio into compact discrete tokens
/// and reconstruct audio from those tokens. They achieve much higher compression ratios
/// than traditional codecs (MP3, AAC) at comparable quality. The tokens can also serve
/// as input to language models for audio generation.
/// </para>
/// <para><b>For Beginners:</b> A neural audio codec is like an AI-powered audio compressor.
/// It converts audio into a very compact code (tokens), then converts that code back to audio.
/// Think of it like a zip file for audio, but using AI instead of traditional compression.
///
/// Two key operations:
/// - Encode: Audio waveform -> compact tokens (small numbers)
/// - Decode: Compact tokens -> reconstructed audio waveform
///
/// Uses:
/// - Ultra-low bitrate audio streaming (1-6 kbps vs 128 kbps for MP3)
/// - Audio tokens for AI language models
/// - High-quality audio compression for storage
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioCodec")]
public interface IAudioCodec<T>
{
    /// <summary>
    /// Gets the sample rate of audio this codec operates on.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the number of quantizer levels (codebooks).
    /// </summary>
    int NumQuantizers { get; }

    /// <summary>
    /// Gets the codebook size (vocabulary size per quantizer).
    /// </summary>
    int CodebookSize { get; }

    /// <summary>
    /// Gets the frame rate of the encoded tokens (tokens per second).
    /// </summary>
    int TokenFrameRate { get; }

    /// <summary>
    /// Encodes audio into discrete tokens.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Encoded tokens as integer indices [num_quantizers, num_frames].</returns>
    int[,] Encode(Tensor<T> audio);

    /// <summary>
    /// Encodes audio asynchronously.
    /// </summary>
    Task<int[,]> EncodeAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Decodes tokens back into audio.
    /// </summary>
    /// <param name="tokens">Encoded tokens [num_quantizers, num_frames].</param>
    /// <returns>Reconstructed audio waveform tensor.</returns>
    Tensor<T> Decode(int[,] tokens);

    /// <summary>
    /// Decodes tokens asynchronously.
    /// </summary>
    Task<Tensor<T>> DecodeAsync(int[,] tokens, CancellationToken cancellationToken = default);

    /// <summary>
    /// Encodes audio into continuous embeddings (before quantization).
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Continuous embedding tensor.</returns>
    Tensor<T> EncodeEmbeddings(Tensor<T> audio);

    /// <summary>
    /// Reconstructs audio from continuous embeddings.
    /// </summary>
    /// <param name="embeddings">Continuous embedding tensor.</param>
    /// <returns>Reconstructed audio waveform.</returns>
    Tensor<T> DecodeEmbeddings(Tensor<T> embeddings);

    /// <summary>
    /// Gets the bitrate at the given number of quantizers in bits per second.
    /// </summary>
    /// <param name="numQuantizers">Number of quantizers to use (null = all).</param>
    /// <returns>Bitrate in bits per second.</returns>
    double GetBitrate(int? numQuantizers = null);
}
