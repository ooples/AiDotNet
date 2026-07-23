namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for multimodal audio-language models that understand and reason about audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio-language models combine audio understanding with natural language processing,
/// enabling tasks like audio captioning, audio question answering, and audio-guided
/// reasoning. They take audio input and text prompts and produce text responses.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio-language models are like ChatGPT but for audio. You can
/// play them a sound and ask questions like "What instruments are playing?" or "Describe
/// the audio scene." They combine the ability to hear (audio encoder) with the ability
/// to understand and respond in natural language (language model).
///
/// How they work:
/// 1. Audio encoder converts sound to features
/// 2. An adapter aligns audio features with the language model
/// 3. The language model processes both audio features and text prompt
/// 4. It generates a text response about the audio
///
/// Common use cases:
/// - Audio captioning: "Describe this sound" -> "A bird singing in a forest"
/// - Audio QA: "What instrument is playing?" -> "A piano"
/// - Audio scene understanding: "Where was this recorded?" -> "An indoor concert hall"
/// - Audio reasoning: "Is this recording happy or sad?" -> "Happy, upbeat tone"
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioLanguageModel")]
public interface IAudioLanguageModel<T>
{
    /// <summary>
    /// Gets the sample rate expected by the audio encoder.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the maximum audio duration in seconds that the model can process.
    /// </summary>
    double MaxAudioDurationSeconds { get; }

    /// <summary>
    /// Gets the maximum number of tokens the model can generate in a response.
    /// </summary>
    int MaxResponseTokens { get; }

    /// <summary>
    /// Gets the list of capabilities this model supports.
    /// </summary>
    /// <returns>A list of capability strings (e.g., "captioning", "qa", "reasoning").</returns>
    IReadOnlyList<string> GetCapabilities();

    /// <summary>
    /// Generates a text response about the given audio based on a text prompt.
    /// </summary>
    /// <param name="audio">The audio waveform to analyze.</param>
    /// <param name="prompt">The text prompt or question about the audio.</param>
    /// <param name="maxTokens">Maximum number of tokens in the response.</param>
    /// <param name="temperature">Sampling temperature (higher = more creative).</param>
    /// <returns>The model's text response about the audio.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method - give it audio and a question, get an answer.
    /// - audio: A recording of someone playing piano
    /// - prompt: "What instrument is being played?"
    /// - returns: "A piano is being played with a gentle, melodic style."
    /// </para>
    /// </remarks>
    string Understand(Tensor<T> audio, string prompt, int maxTokens = 256, double temperature = 0.7);

    /// <summary>
    /// Generates a text response about audio asynchronously.
    /// </summary>
    Task<string> UnderstandAsync(Tensor<T> audio, string prompt, int maxTokens = 256,
        double temperature = 0.7, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a caption describing the audio content.
    /// </summary>
    /// <param name="audio">The audio waveform to caption.</param>
    /// <param name="maxTokens">Maximum caption length in tokens.</param>
    /// <returns>A text description of the audio content.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Like image captioning but for audio.
    /// - Input: Recording of a thunderstorm
    /// - Output: "Heavy rain falling with distant thunder and occasional lightning strikes."
    /// </para>
    /// </remarks>
    string Caption(Tensor<T> audio, int maxTokens = 128);

    /// <summary>
    /// Extracts audio embeddings that can be used for downstream tasks.
    /// </summary>
    /// <param name="audio">The audio waveform.</param>
    /// <returns>Audio embedding tensor.</returns>
    Tensor<T> ExtractAudioEmbeddings(Tensor<T> audio);
}
