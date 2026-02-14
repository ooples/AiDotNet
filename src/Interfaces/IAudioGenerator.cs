namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for audio generation models that create audio from text descriptions or other conditions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio generation models create sounds, music, and audio effects from various inputs.
/// Unlike TTS which focuses on speech, audio generators can produce any type of sound
/// including music, environmental sounds, and sound effects.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio generation is like having an artist who can create
/// any sound you describe.
///
/// How audio generation works:
/// 1. You provide a description ("A dog barking in a park")
/// 2. The model generates audio features that match the description
/// 3. The features are converted to playable audio
///
/// Types of audio generation:
/// - Text-to-Audio: "Thunder during a storm" creates thunder sounds
/// - Text-to-Music: "Upbeat jazz piano" creates music
/// - Audio Inpainting: Fill in missing parts of audio
/// - Audio Continuation: Extend existing audio naturally
///
/// Common use cases:
/// - Video game sound effects
/// - Film and media production
/// - Music composition assistance
/// - Podcast and content creation
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioGenerator")]
public interface IAudioGenerator<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the sample rate of generated audio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 16000 Hz (low quality), 22050 Hz (medium), 44100 Hz (CD quality).
    /// </para>
    /// </remarks>
    int SampleRate { get; }

    /// <summary>
    /// Gets the maximum duration of audio that can be generated in seconds.
    /// </summary>
    double MaxDurationSeconds { get; }

    /// <summary>
    /// Gets whether this model supports text-to-audio generation.
    /// </summary>
    bool SupportsTextToAudio { get; }

    /// <summary>
    /// Gets whether this model supports text-to-music generation.
    /// </summary>
    bool SupportsTextToMusic { get; }

    /// <summary>
    /// Gets whether this model supports audio continuation.
    /// </summary>
    bool SupportsAudioContinuation { get; }

    /// <summary>
    /// Gets whether this model supports audio inpainting.
    /// </summary>
    bool SupportsAudioInpainting { get; }

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
    /// Generates audio from a text description.
    /// </summary>
    /// <param name="prompt">Text description of the desired audio.</param>
    /// <param name="negativePrompt">What to avoid in the generated audio.</param>
    /// <param name="durationSeconds">Length of audio to generate.</param>
    /// <param name="numInferenceSteps">Number of generation steps (more = higher quality).</param>
    /// <param name="guidanceScale">How closely to follow the prompt (higher = more literal).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Generated audio waveform tensor [samples] or [channels, samples].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates sound effects or ambient audio from descriptions.
    /// - prompt: "Ocean waves crashing on a beach" creates wave sounds
    /// - prompt: "Birds chirping in a forest" creates bird sounds
    /// - negativePrompt: "No human voices" prevents speech in the output
    /// </para>
    /// </remarks>
    Tensor<T> GenerateAudio(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Generates audio from a text description asynchronously.
    /// </summary>
    /// <param name="prompt">Text description of the desired audio.</param>
    /// <param name="negativePrompt">What to avoid in the generated audio.</param>
    /// <param name="durationSeconds">Length of audio to generate.</param>
    /// <param name="numInferenceSteps">Number of generation steps.</param>
    /// <param name="guidanceScale">How closely to follow the prompt.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Generated audio waveform tensor.</returns>
    Task<Tensor<T>> GenerateAudioAsync(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates music from a text description.
    /// </summary>
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="negativePrompt">What to avoid in the generated music.</param>
    /// <param name="durationSeconds">Length of music to generate.</param>
    /// <param name="numInferenceSteps">Number of generation steps.</param>
    /// <param name="guidanceScale">How closely to follow the prompt.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Generated music waveform tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates music from descriptions.
    /// - prompt: "Relaxing piano melody" creates piano music
    /// - prompt: "Energetic rock guitar riff" creates rock music
    /// </para>
    /// </remarks>
    Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 10.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null);

    /// <summary>
    /// Continues existing audio to extend it naturally.
    /// </summary>
    /// <param name="inputAudio">The audio to continue from.</param>
    /// <param name="prompt">Optional text guidance for continuation.</param>
    /// <param name="extensionSeconds">How many seconds to add.</param>
    /// <param name="numInferenceSteps">Number of generation steps.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Extended audio waveform (original + continuation).</returns>
    /// <exception cref="NotSupportedException">Thrown if continuation is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This extends audio by generating more that follows naturally.
    /// - Input: 5 seconds of guitar
    /// - Output: Original + 10 more seconds in the same style
    /// </para>
    /// </remarks>
    Tensor<T> ContinueAudio(
        Tensor<T> inputAudio,
        string? prompt = null,
        double extensionSeconds = 5.0,
        int numInferenceSteps = 100,
        int? seed = null);

    /// <summary>
    /// Fills in missing or masked sections of audio.
    /// </summary>
    /// <param name="audio">Audio with sections to fill.</param>
    /// <param name="mask">Mask tensor indicating which samples to regenerate (1 = regenerate, 0 = keep).</param>
    /// <param name="prompt">Optional text guidance for inpainting.</param>
    /// <param name="numInferenceSteps">Number of generation steps.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Audio with masked sections filled in.</returns>
    /// <exception cref="NotSupportedException">Thrown if inpainting is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This fills in gaps in audio, like photo inpainting but for sound.
    /// - Input: Audio with a 2-second gap (maybe someone coughed)
    /// - Output: Audio with the gap filled seamlessly
    /// </para>
    /// </remarks>
    Tensor<T> InpaintAudio(
        Tensor<T> audio,
        Tensor<T> mask,
        string? prompt = null,
        int numInferenceSteps = 100,
        int? seed = null);

    /// <summary>
    /// Gets generation options for advanced control.
    /// </summary>
    AudioGenerationOptions<T> GetDefaultOptions();
}

/// <summary>
/// Advanced options for audio generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AudioGenerationOptions<T>
{
    /// <summary>
    /// Gets or sets the duration of audio to generate in seconds.
    /// </summary>
    public double DurationSeconds { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets the number of inference steps.
    /// </summary>
    public int NumInferenceSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the guidance scale.
    /// </summary>
    public double GuidanceScale { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the random seed.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to generate stereo audio.
    /// </summary>
    public bool Stereo { get; set; } = false;

    /// <summary>
    /// Gets or sets the scheduler type for the diffusion process.
    /// </summary>
    public string SchedulerType { get; set; } = "ddpm";
}
