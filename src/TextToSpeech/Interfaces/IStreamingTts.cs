namespace AiDotNet.TextToSpeech.Interfaces;

/// <summary>
/// Interface for TTS models that support streaming/chunked synthesis with low latency.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Streaming TTS models can begin outputting audio before the full utterance is processed,
/// enabling low-latency applications like conversational AI:
/// <list type="bullet">
/// <item>CosyVoice 2: 150ms first-packet latency with streaming flow matching</item>
/// <item>Chatterbox: real-time streaming with emotion control</item>
/// <item>XTTS-v2: chunked streaming for voice cloning</item>
/// </list>
/// </para>
/// </remarks>
public interface IStreamingTts<T> : ITtsModel<T>
{
    /// <summary>
    /// Synthesizes audio in streaming chunks, returning the first available audio chunk.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <param name="chunkSize">Number of audio samples per chunk.</param>
    /// <returns>First audio chunk tensor.</returns>
    Tensor<T> SynthesizeFirstChunk(string text, int chunkSize);

    /// <summary>
    /// Gets the next audio chunk from an ongoing streaming synthesis.
    /// </summary>
    /// <returns>Next audio chunk tensor, or empty tensor if synthesis is complete.</returns>
    Tensor<T> SynthesizeNextChunk();

    /// <summary>
    /// Gets the target first-packet latency in milliseconds.
    /// </summary>
    int FirstPacketLatencyMs { get; }

    /// <summary>
    /// Gets whether there are more audio chunks available.
    /// </summary>
    bool HasMoreChunks { get; }
}
