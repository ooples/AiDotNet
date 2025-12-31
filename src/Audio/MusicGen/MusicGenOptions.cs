using AiDotNet.Onnx;

namespace AiDotNet.Audio.MusicGen;

/// <summary>
/// Configuration options for MusicGen text-to-music generation.
/// </summary>
/// <remarks>
/// <para>
/// MusicGen is Meta's state-of-the-art music generation model that creates
/// high-quality music from text descriptions. It uses a single-stage transformer
/// language model operating over EnCodec audio codes.
/// </para>
/// <para><b>For Beginners:</b> MusicGen generates actual music from descriptions:
///
/// Example prompts:
/// - "Upbeat electronic dance music with heavy bass"
/// - "Calm acoustic guitar melody with soft drums"
/// - "Epic orchestral piece with dramatic strings"
/// - "Lo-fi hip hop beats for studying"
///
/// Tips for good prompts:
/// - Be specific about genre, instruments, and mood
/// - Include tempo hints (fast, slow, moderate)
/// - Mention energy level (energetic, calm, building)
/// </para>
/// </remarks>
public class MusicGenOptions
{
    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <remarks>
    /// Different sizes trade off quality vs speed.
    /// Default is Medium which balances both well.
    /// </remarks>
    public MusicGenModelSize ModelSize { get; set; } = MusicGenModelSize.Medium;

    /// <summary>
    /// Gets or sets the output sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// MusicGen uses 32kHz by default, matching the EnCodec codec.
    /// This produces high-quality audio suitable for music.
    /// </remarks>
    public int SampleRate { get; set; } = 32000;

    /// <summary>
    /// Gets or sets the default duration of generated music in seconds.
    /// </summary>
    public double DurationSeconds { get; set; } = 8.0;

    /// <summary>
    /// Gets or sets the maximum duration in seconds.
    /// </summary>
    /// <remarks>
    /// MusicGen can generate up to 30 seconds of audio.
    /// Longer durations require more memory and compute time.
    /// </remarks>
    public double MaxDurationSeconds { get; set; } = 30.0;

    /// <summary>
    /// Gets or sets the sampling temperature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls randomness in generation:
    /// - Lower (0.5-0.8): More predictable, stable output
    /// - Default (1.0): Balanced creativity
    /// - Higher (1.2-2.0): More creative but potentially less coherent
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the top-k sampling parameter.
    /// </summary>
    /// <remarks>
    /// Limits sampling to the top K most likely tokens.
    /// Default of 250 works well for diverse music generation.
    /// </remarks>
    public int TopK { get; set; } = 250;

    /// <summary>
    /// Gets or sets the top-p (nucleus) sampling parameter.
    /// </summary>
    /// <remarks>
    /// If greater than 0, uses nucleus sampling instead of top-k.
    /// Value of 0.0 disables nucleus sampling (uses top-k only).
    /// </remarks>
    public double TopP { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the classifier-free guidance scale.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls how closely the model follows the text prompt:
    /// - Low (1.0-2.0): More variation, less prompt adherence
    /// - Default (3.0): Good balance
    /// - High (4.0-7.0): Stricter prompt following, less creativity
    /// </para>
    /// </remarks>
    public double GuidanceScale { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// Set to a specific value to generate the same music each time.
    /// Null for random generation.
    /// </remarks>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to generate stereo audio.
    /// </summary>
    /// <remarks>
    /// Requires the Stereo model variant for best results.
    /// When using non-stereo models, mono output is duplicated to stereo.
    /// </remarks>
    public bool Stereo { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of EnCodec codebooks to use.
    /// </summary>
    /// <remarks>
    /// More codebooks = higher quality but slower generation.
    /// Default of 4 is used by standard MusicGen models.
    /// </remarks>
    public int NumCodebooks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the codebook vocabulary size.
    /// </summary>
    /// <remarks>
    /// Must match the EnCodec model configuration.
    /// Default of 2048 is standard for MusicGen.
    /// </remarks>
    public int CodebookSize { get; set; } = 2048;

    /// <summary>
    /// Gets or sets whether to use the delay pattern.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The delay pattern is MusicGen's key innovation:
    /// - Generates codebooks with temporal offset
    /// - Reduces effective sequence length
    /// - Improves generation efficiency
    /// </para>
    /// Should be true for standard MusicGen operation.
    /// </remarks>
    public bool UseDelayPattern { get; set; } = true;

    /// <summary>
    /// Gets or sets the path to the text encoder ONNX model.
    /// </summary>
    public string? TextEncoderPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the language model ONNX model.
    /// </summary>
    public string? LanguageModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the EnCodec decoder ONNX model.
    /// </summary>
    public string? EnCodecDecoderPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX execution options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the dropout rate for training.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum text sequence length.
    /// </summary>
    public int MaxTextLength { get; set; } = 256;
}
