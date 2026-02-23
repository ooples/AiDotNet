using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.AudioGen;

/// <summary>
/// Configuration options for audio generation models.
/// </summary>
/// <remarks>
/// <para>
/// AudioGen models generate audio from text descriptions using a language model
/// approach with discrete audio codes (like EnCodec).
/// </para>
/// <para><b>For Beginners:</b> AudioGen works differently from TTS:
/// - TTS: Converts specific text to spoken words
/// - AudioGen: Creates sounds/music matching a description
///
/// Example prompts:
/// - "A dog barking in the distance"
/// - "Gentle piano music with rain sounds"
/// - "Crowd cheering at a sports event"
/// </para>
/// </remarks>
public class AudioGenOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public AudioGenOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AudioGenOptions(AudioGenOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ModelSize = other.ModelSize;
        SampleRate = other.SampleRate;
        DurationSeconds = other.DurationSeconds;
        MaxDurationSeconds = other.MaxDurationSeconds;
        Temperature = other.Temperature;
        TopK = other.TopK;
        TopP = other.TopP;
        GuidanceScale = other.GuidanceScale;
        Channels = other.Channels;
        TextEncoderPath = other.TextEncoderPath;
        LanguageModelPath = other.LanguageModelPath;
        AudioCodecPath = other.AudioCodecPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>
    /// Gets or sets the model size to use.
    /// </summary>
    public AudioGenModelSize ModelSize { get; set; } = AudioGenModelSize.Medium;

    /// <summary>
    /// Gets or sets the output sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 32000;

    /// <summary>
    /// Gets or sets the duration of generated audio in seconds.
    /// </summary>
    public double DurationSeconds { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets the maximum duration in seconds.
    /// </summary>
    public double MaxDurationSeconds { get; set; } = 30.0;

    /// <summary>
    /// Gets or sets the temperature for sampling.
    /// Higher values = more random, lower = more deterministic.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the top-k value for sampling.
    /// Only the top k tokens are considered.
    /// </summary>
    public int TopK { get; set; } = 250;

    /// <summary>
    /// Gets or sets the top-p (nucleus) value for sampling.
    /// </summary>
    public double TopP { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the classifier-free guidance scale.
    /// Higher values = stronger prompt following.
    /// </summary>
    public double GuidanceScale { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// Null for random generation.
    /// </summary>
    public new int? Seed { get; set; }

    /// <summary>
    /// Gets or sets the number of audio channels (1=mono, 2=stereo).
    /// </summary>
    public int Channels { get; set; } = 1;

    /// <summary>
    /// Gets or sets the path to the text encoder model.
    /// </summary>
    public string? TextEncoderPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the language model.
    /// </summary>
    public string? LanguageModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the audio codec (decoder) model.
    /// </summary>
    public string? AudioCodecPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX execution options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
