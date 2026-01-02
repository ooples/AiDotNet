namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for speech recognition models that transcribe audio to text (ASR - Automatic Speech Recognition).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speech recognition models convert spoken audio into written text. They analyze audio waveforms
/// or spectrograms to identify phonemes, words, and sentences. Modern speech recognition uses
/// encoder-decoder architectures (like Whisper) or CTC-based models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Speech recognition is like having a transcriptionist listen to audio
/// and type out what they hear.
///
/// How speech recognition works:
/// 1. Audio is converted to features (spectrograms or mel-spectrograms)
/// 2. The model processes these features to identify speech patterns
/// 3. Patterns are decoded into words and sentences
///
/// Common use cases:
/// - Voice assistants (Siri, Alexa, Google Assistant)
/// - Video/podcast transcription
/// - Real-time captioning for accessibility
/// - Voice typing and dictation
///
/// Key challenges:
/// - Different accents and speaking styles
/// - Background noise and multiple speakers
/// - Domain-specific vocabulary (medical, legal terms)
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
public interface ISpeechRecognizer<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the sample rate expected by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Most speech recognition models expect 16000 Hz audio. Input audio should be
    /// resampled to match this rate before processing.
    /// </para>
    /// </remarks>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of languages supported by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multilingual models like Whisper support many languages. Monolingual models
    /// may only support one. Check this property before processing foreign audio.
    /// </para>
    /// </remarks>
    IReadOnlyList<string> SupportedLanguages { get; }

    /// <summary>
    /// Gets whether this model supports real-time streaming transcription.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Streaming mode transcribes audio as it comes in,
    /// without waiting for the entire recording. Good for live captioning.
    /// </para>
    /// </remarks>
    bool SupportsStreaming { get; }

    /// <summary>
    /// Gets whether this model can identify timestamps for each word.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Word-level timestamps are useful for subtitle generation and audio editing.
    /// </para>
    /// </remarks>
    bool SupportsWordTimestamps { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses pre-trained ONNX weights for inference.
    /// When false, the model can be trained from scratch using the neural network infrastructure.
    /// </para>
    /// </remarks>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Transcribes audio to text.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [batch, samples] or [samples].</param>
    /// <param name="language">Optional language code (e.g., "en", "es"). Auto-detected if null.</param>
    /// <param name="includeTimestamps">Whether to include word-level timestamps.</param>
    /// <returns>Transcription result containing text and optional timestamps.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for converting speech to text.
    /// - Pass in audio data (as a tensor of samples)
    /// - Get back the transcribed text
    /// </para>
    /// </remarks>
    TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false);

    /// <summary>
    /// Transcribes audio to text asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [batch, samples] or [samples].</param>
    /// <param name="language">Optional language code (e.g., "en", "es"). Auto-detected if null.</param>
    /// <param name="includeTimestamps">Whether to include word-level timestamps.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Transcription result containing text and optional timestamps.</returns>
    Task<TranscriptionResult<T>> TranscribeAsync(
        Tensor<T> audio,
        string? language = null,
        bool includeTimestamps = false,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Detects the language spoken in the audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [batch, samples] or [samples].</param>
    /// <returns>Detected language code (e.g., "en", "es", "fr").</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This identifies what language is being spoken
    /// before transcription. Useful for multilingual applications.
    /// </para>
    /// </remarks>
    string DetectLanguage(Tensor<T> audio);

    /// <summary>
    /// Gets language detection probabilities for the audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [batch, samples] or [samples].</param>
    /// <returns>Dictionary mapping language codes to confidence scores (0.0 to 1.0).</returns>
    IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio);

    /// <summary>
    /// Starts a streaming transcription session.
    /// </summary>
    /// <param name="language">Optional language code for transcription.</param>
    /// <returns>A streaming session that can receive audio chunks incrementally.</returns>
    /// <exception cref="NotSupportedException">Thrown if streaming is not supported.</exception>
    IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null);
}

/// <summary>
/// Represents the result of a transcription operation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TranscriptionResult<T>
{
    /// <summary>
    /// Gets or sets the transcribed text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the detected or specified language.
    /// </summary>
    public string Language { get; set; } = "en";

    /// <summary>
    /// Gets or sets the confidence score for the transcription (0.0 to 1.0).
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the word-level segments with timestamps.
    /// </summary>
    public IReadOnlyList<TranscriptionSegment<T>> Segments { get; set; } = Array.Empty<TranscriptionSegment<T>>();

    /// <summary>
    /// Gets or sets the total duration of the audio in seconds.
    /// </summary>
    public double DurationSeconds { get; set; }
}

/// <summary>
/// Represents a segment of transcribed text with timing information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TranscriptionSegment<T>
{
    /// <summary>
    /// Gets or sets the text content of this segment.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the confidence score for this segment.
    /// </summary>
    public T Confidence { get; set; } = default!;
}

/// <summary>
/// Interface for streaming transcription sessions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IStreamingTranscriptionSession<T> : IDisposable
{
    /// <summary>
    /// Feeds an audio chunk to the streaming session.
    /// </summary>
    /// <param name="audioChunk">Audio samples to process.</param>
    void FeedAudio(Tensor<T> audioChunk);

    /// <summary>
    /// Gets the current partial transcription.
    /// </summary>
    /// <returns>The current partial result.</returns>
    TranscriptionResult<T> GetPartialResult();

    /// <summary>
    /// Finalizes the session and returns the complete transcription.
    /// </summary>
    /// <returns>The final transcription result.</returns>
    TranscriptionResult<T> Finalize();
}
