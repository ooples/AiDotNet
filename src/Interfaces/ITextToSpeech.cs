namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for text-to-speech (TTS) models that synthesize spoken audio from text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Text-to-speech models convert written text into natural-sounding spoken audio.
/// Modern TTS systems use neural networks to produce high-quality, expressive speech
/// that can sound nearly indistinguishable from human speakers.
/// </para>
/// <para>
/// <b>For Beginners:</b> TTS is like having a computer read text out loud to you.
///
/// How TTS works:
/// 1. Text is analyzed for pronunciation, emphasis, and pacing
/// 2. The model generates audio features (mel-spectrograms)
/// 3. A vocoder converts features to waveform audio
///
/// Common use cases:
/// - Accessibility (screen readers for visually impaired)
/// - Voice assistants and chatbots
/// - Audiobook and podcast generation
/// - Language learning applications
///
/// Key features:
/// - Voice cloning: Make it sound like a specific person
/// - Emotion control: Express happiness, sadness, excitement
/// - Speed control: Speak faster or slower
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
public interface ITextToSpeech<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the sample rate of generated audio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 22050 Hz (standard), 44100 Hz (high quality), 16000 Hz (telephony).
    /// </para>
    /// </remarks>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of available built-in voices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each voice has unique characteristics (gender, age, accent, style).
    /// </para>
    /// </remarks>
    IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; }

    /// <summary>
    /// Gets whether this model supports voice cloning from reference audio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Voice cloning lets you make the TTS sound like
    /// a specific person by providing a sample of their voice.
    /// </para>
    /// </remarks>
    bool SupportsVoiceCloning { get; }

    /// <summary>
    /// Gets whether this model supports emotional expression control.
    /// </summary>
    bool SupportsEmotionControl { get; }

    /// <summary>
    /// Gets whether this model supports streaming audio generation.
    /// </summary>
    bool SupportsStreaming { get; }

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
    /// Synthesizes speech from text.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="voiceId">Optional voice identifier. Uses default if null.</param>
    /// <param name="speakingRate">Speed multiplier (0.5 = half speed, 2.0 = double speed).</param>
    /// <param name="pitch">Pitch adjustment in semitones (-12 to +12).</param>
    /// <returns>Audio waveform tensor [samples] or [channels, samples].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for converting text to speech.
    /// - Pass in text like "Hello, how are you?"
    /// - Get back audio you can play through speakers
    /// </para>
    /// </remarks>
    Tensor<T> Synthesize(
        string text,
        string? voiceId = null,
        double speakingRate = 1.0,
        double pitch = 0.0);

    /// <summary>
    /// Synthesizes speech from text asynchronously.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="voiceId">Optional voice identifier. Uses default if null.</param>
    /// <param name="speakingRate">Speed multiplier (0.5 = half speed, 2.0 = double speed).</param>
    /// <param name="pitch">Pitch adjustment in semitones (-12 to +12).</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Audio waveform tensor [samples] or [channels, samples].</returns>
    Task<Tensor<T>> SynthesizeAsync(
        string text,
        string? voiceId = null,
        double speakingRate = 1.0,
        double pitch = 0.0,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synthesizes speech using a cloned voice from reference audio.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="referenceAudio">Reference audio sample of the voice to clone.</param>
    /// <param name="speakingRate">Speed multiplier.</param>
    /// <param name="pitch">Pitch adjustment in semitones.</param>
    /// <returns>Audio waveform tensor matching the reference voice.</returns>
    /// <exception cref="NotSupportedException">Thrown if voice cloning is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates speech that sounds like the person
    /// in the reference audio. The model learns the voice characteristics
    /// from the sample and applies them to new text.
    /// </para>
    /// </remarks>
    Tensor<T> SynthesizeWithVoiceCloning(
        string text,
        Tensor<T> referenceAudio,
        double speakingRate = 1.0,
        double pitch = 0.0);

    /// <summary>
    /// Synthesizes speech with emotional expression.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="emotion">The emotion to express (e.g., "happy", "sad", "angry").</param>
    /// <param name="emotionIntensity">Intensity of the emotion (0.0 to 1.0).</param>
    /// <param name="voiceId">Optional voice identifier.</param>
    /// <param name="speakingRate">Speed multiplier.</param>
    /// <returns>Audio waveform tensor with emotional expression.</returns>
    /// <exception cref="NotSupportedException">Thrown if emotion control is not supported.</exception>
    Tensor<T> SynthesizeWithEmotion(
        string text,
        string emotion,
        double emotionIntensity = 0.5,
        string? voiceId = null,
        double speakingRate = 1.0);

    /// <summary>
    /// Extracts speaker embedding from reference audio for voice cloning.
    /// </summary>
    /// <param name="referenceAudio">Reference audio sample.</param>
    /// <returns>Speaker embedding tensor that captures voice characteristics.</returns>
    Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio);

    /// <summary>
    /// Starts a streaming synthesis session for incremental audio generation.
    /// </summary>
    /// <param name="voiceId">Optional voice identifier.</param>
    /// <param name="speakingRate">Speed multiplier.</param>
    /// <returns>A streaming session that can receive text incrementally.</returns>
    /// <exception cref="NotSupportedException">Thrown if streaming is not supported.</exception>
    IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0);
}

/// <summary>
/// Information about an available TTS voice.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VoiceInfo<T>
{
    /// <summary>
    /// Gets or sets the unique identifier for this voice.
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the display name of this voice.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the language code for this voice.
    /// </summary>
    public string Language { get; set; } = "en";

    /// <summary>
    /// Gets or sets the gender of this voice.
    /// </summary>
    public VoiceGender Gender { get; set; } = VoiceGender.Neutral;

    /// <summary>
    /// Gets or sets the style description of this voice.
    /// </summary>
    public string Style { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the pre-computed speaker embedding for this voice.
    /// </summary>
    public Tensor<T>? SpeakerEmbedding { get; set; }
}

/// <summary>
/// Gender classification for TTS voices.
/// </summary>
public enum VoiceGender
{
    /// <summary>Neutral or unspecified gender.</summary>
    Neutral,
    /// <summary>Male voice.</summary>
    Male,
    /// <summary>Female voice.</summary>
    Female
}

/// <summary>
/// Interface for streaming TTS synthesis sessions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IStreamingSynthesisSession<T> : IDisposable
{
    /// <summary>
    /// Feeds text to the streaming session.
    /// </summary>
    /// <param name="textChunk">Text chunk to synthesize.</param>
    void FeedText(string textChunk);

    /// <summary>
    /// Gets available audio chunks that have been synthesized.
    /// </summary>
    /// <returns>Audio chunks ready for playback.</returns>
    IEnumerable<Tensor<T>> GetAvailableAudio();

    /// <summary>
    /// Finalizes the session and returns any remaining audio.
    /// </summary>
    /// <returns>Final audio chunks.</returns>
    IEnumerable<Tensor<T>> Finalize();
}
