namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for audio diffusion models that generate sound and music.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio diffusion models apply diffusion processes to generate audio content,
/// including music, speech, sound effects, and more. They typically operate on
/// audio spectrograms or mel-spectrograms in latent space.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio diffusion models work similarly to image diffusion,
/// but instead of generating pictures, they create sounds.
///
/// How audio diffusion works:
/// 1. Audio is converted to a spectrogram (visual representation of sound)
/// 2. Diffusion happens on this spectrogram (just like image diffusion)
/// 3. The spectrogram is converted back to audio
///
/// Types of audio generation:
/// - Text-to-Audio: "A dog barking in a park" → audio clip
/// - Text-to-Music: "Upbeat jazz piano" → music track
/// - Text-to-Speech: Text → spoken voice
/// - Audio-to-Audio: Transform existing audio (voice conversion, style transfer)
///
/// Key challenges:
/// - Temporal coherence (sounds must flow naturally)
/// - Frequency relationships (harmonics, rhythm)
/// - Long-range dependencies (verse-chorus structure in music)
/// </para>
/// <para>
/// This interface extends <see cref="IDiffusionModel{T}"/> with audio-specific operations.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioDiffusionModel")]
public interface IAudioDiffusionModel<T> : IDiffusionModel<T>
{
    /// <summary>
    /// Gets the sample rate of generated audio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 16000 Hz (speech), 22050 Hz (music), 44100 Hz (high quality).
    /// Higher sample rates = better quality but more computation.
    /// </para>
    /// </remarks>
    int SampleRate { get; }

    /// <summary>
    /// Gets the default duration of generated audio in seconds.
    /// </summary>
    double DefaultDurationSeconds { get; }

    /// <summary>
    /// Gets whether this model supports text-to-audio generation.
    /// </summary>
    bool SupportsTextToAudio { get; }

    /// <summary>
    /// Gets whether this model supports text-to-music generation.
    /// </summary>
    bool SupportsTextToMusic { get; }

    /// <summary>
    /// Gets whether this model supports text-to-speech generation.
    /// </summary>
    bool SupportsTextToSpeech { get; }

    /// <summary>
    /// Gets whether this model supports audio-to-audio transformation.
    /// </summary>
    bool SupportsAudioToAudio { get; }

    /// <summary>
    /// Gets the number of mel spectrogram channels used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mel spectrograms divide the frequency range into perceptual bands.
    /// Common values: 64, 80, or 128 mel bins.
    /// </para>
    /// </remarks>
    int MelChannels { get; }

    /// <summary>
    /// Generates audio from a text description.
    /// </summary>
    /// <param name="prompt">Text description of the desired audio.</param>
    /// <param name="negativePrompt">What to avoid in the audio.</param>
    /// <param name="durationSeconds">Length of audio to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor [batch, samples] or [batch, channels, samples] for stereo.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates sound from a description:
    /// - prompt: "A thunderstorm with rain" → Thunder and rain sounds
    /// - prompt: "Acoustic guitar strumming" → Guitar music
    /// </para>
    /// </remarks>
    Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Generates music from a text description.
    /// </summary>
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="negativePrompt">What to avoid in the music.</param>
    /// <param name="durationSeconds">Length of music to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor.</returns>
    /// <remarks>
    /// <para>
    /// Music generation may use specialized models tuned for musical content,
    /// with better handling of melody, harmony, and rhythm.
    /// </para>
    /// </remarks>
    Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Synthesizes speech from text (text-to-speech).
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="speakerEmbedding">Optional speaker embedding for voice cloning.</param>
    /// <param name="speakingRate">Speed multiplier (1.0 = normal).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This makes the computer "read" text out loud:
    /// - Input: "Hello, how are you today?"
    /// - Output: Audio of someone saying those words
    /// - speakerEmbedding: Makes it sound like a specific person
    /// </para>
    /// </remarks>
    Tensor<T> TextToSpeech(
        string text,
        Tensor<T>? speakerEmbedding = null,
        double speakingRate = 1.0,
        int numInferenceSteps = 50,
        int? seed = null);

    /// <summary>
    /// Transforms existing audio based on a text prompt.
    /// </summary>
    /// <param name="inputAudio">The input audio waveform.</param>
    /// <param name="prompt">Text description of the transformation.</param>
    /// <param name="negativePrompt">What to avoid.</param>
    /// <param name="strength">Transformation strength (0.0-1.0).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Transformed audio waveform.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This changes existing audio:
    /// - "Make it sound like it's underwater"
    /// - "Add reverb like a large hall"
    /// - "Change the voice to sound younger"
    /// </para>
    /// </remarks>
    Tensor<T> AudioToAudio(
        Tensor<T> inputAudio,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Continues/extends audio from a given clip.
    /// </summary>
    /// <param name="inputAudio">The audio to continue from.</param>
    /// <param name="prompt">Optional text guidance for continuation.</param>
    /// <param name="extensionSeconds">How many seconds to add.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Extended audio waveform (original + continuation).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This extends audio by generating more that follows:
    /// - Input: 5 seconds of a song
    /// - Output: Original 5 seconds + 10 more seconds that fit naturally
    /// </para>
    /// </remarks>
    Tensor<T> ContinueAudio(
        Tensor<T> inputAudio,
        string? prompt = null,
        double extensionSeconds = 5.0,
        int numInferenceSteps = 100,
        int? seed = null);

    /// <summary>
    /// Converts audio waveform to mel spectrogram.
    /// </summary>
    /// <param name="waveform">Audio waveform [batch, samples].</param>
    /// <returns>Mel spectrogram [batch, channels, melBins, timeFrames].</returns>
    Tensor<T> WaveformToMelSpectrogram(Tensor<T> waveform);

    /// <summary>
    /// Converts mel spectrogram back to audio waveform.
    /// </summary>
    /// <param name="melSpectrogram">Mel spectrogram [batch, channels, melBins, timeFrames].</param>
    /// <returns>Audio waveform [batch, samples].</returns>
    Tensor<T> MelSpectrogramToWaveform(Tensor<T> melSpectrogram);

    /// <summary>
    /// Gets speaker embeddings from a reference audio clip (for voice cloning).
    /// </summary>
    /// <param name="referenceAudio">Reference audio waveform.</param>
    /// <returns>Speaker embedding tensor.</returns>
    Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio);
}
