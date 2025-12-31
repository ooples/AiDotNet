using AiDotNet.Onnx;

namespace AiDotNet.Audio.StableAudio;

/// <summary>
/// Configuration options for Stable Audio generation.
/// </summary>
/// <remarks>
/// <para>
/// Stable Audio is Stability AI's state-of-the-art audio generation model using
/// latent diffusion with a Diffusion Transformer (DiT) architecture. It supports
/// high-quality music and sound effects generation with variable-length output.
/// </para>
/// <para><b>For Beginners:</b> Stable Audio generates professional-quality audio:
///
/// Example prompts:
/// - "Upbeat electronic dance track with synth leads and heavy bass drop"
/// - "Peaceful ambient soundscape with soft pads and nature sounds"
/// - "Epic orchestral trailer music with dramatic brass and percussion"
/// - "Lo-fi hip hop beat with jazzy piano chords and vinyl crackle"
///
/// Tips for good prompts:
/// - Be specific about genre, instruments, mood, and tempo
/// - Mention audio characteristics (stereo width, dynamics)
/// - Include style references when appropriate
/// </para>
/// </remarks>
public class StableAudioOptions
{
    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <remarks>
    /// Different sizes trade off quality vs speed.
    /// Default is Base which balances both well.
    /// </remarks>
    public StableAudioModelSize ModelSize { get; set; } = StableAudioModelSize.Base;

    /// <summary>
    /// Gets or sets the output sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// Stable Audio uses 44.1kHz by default for CD-quality audio.
    /// This is the professional music standard sample rate.
    /// </remarks>
    public int SampleRate { get; set; } = 44100;

    /// <summary>
    /// Gets or sets the default duration of generated audio in seconds.
    /// </summary>
    public double DurationSeconds { get; set; } = 30.0;

    /// <summary>
    /// Gets or sets the maximum duration in seconds.
    /// </summary>
    /// <remarks>
    /// Stable Audio 2.0 can generate up to 180 seconds (3 minutes) of audio.
    /// The Open variant supports up to 47 seconds.
    /// Longer durations require more memory and compute time.
    /// </remarks>
    public double MaxDurationSeconds { get; set; } = 180.0;

    /// <summary>
    /// Gets or sets the number of diffusion steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More steps = higher quality but slower generation:
    /// - 25 steps: Fast, lower quality
    /// - 50 steps: Good balance
    /// - 100 steps: High quality (default)
    /// - 200+ steps: Best quality, slow
    /// </para>
    /// </remarks>
    public int NumInferenceSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the classifier-free guidance scale.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls how closely the model follows the text prompt:
    /// - Low (1.0-3.0): More variation, less prompt adherence
    /// - Default (7.0): Good balance
    /// - High (10.0-15.0): Stricter prompt following, may reduce quality
    /// </para>
    /// </remarks>
    public double GuidanceScale { get; set; } = 7.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// Set to a specific value to generate the same audio each time.
    /// Null for random generation.
    /// </remarks>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to generate stereo audio.
    /// </summary>
    /// <remarks>
    /// When true, generates two-channel stereo output.
    /// Stable Audio natively supports stereo generation.
    /// </remarks>
    public bool Stereo { get; set; } = true;

    /// <summary>
    /// Gets or sets the latent dimension.
    /// </summary>
    /// <remarks>
    /// The dimension of the compressed audio representation.
    /// Default of 64 matches standard Stable Audio architecture.
    /// </remarks>
    public int LatentDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the DiT hidden dimension.
    /// </summary>
    /// <remarks>
    /// Hidden dimension of the Diffusion Transformer blocks.
    /// Default of 1024 is for Base model.
    /// </remarks>
    public int DitHiddenDim { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of DiT blocks.
    /// </summary>
    /// <remarks>
    /// Number of Diffusion Transformer blocks.
    /// More blocks = more capacity but slower.
    /// </remarks>
    public int NumDitBlocks { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumAttentionHeads { get; set; } = 16;

    /// <summary>
    /// Gets or sets the path to the T5 text encoder ONNX model.
    /// </summary>
    public string? TextEncoderPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the VAE ONNX model.
    /// </summary>
    public string? VaePath { get; set; }

    /// <summary>
    /// Gets or sets the path to the DiT denoiser ONNX model.
    /// </summary>
    public string? DitPath { get; set; }

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
    public int MaxTextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the maximum audio latent length.
    /// </summary>
    public int MaxAudioLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the T5 embedding dimension.
    /// </summary>
    public int TextEmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the conditioning scale for timing information.
    /// </summary>
    /// <remarks>
    /// Stable Audio uses duration and timing conditioning.
    /// This controls how strongly the model follows timing information.
    /// </remarks>
    public double TimingConditioningScale { get; set; } = 1.0;
}
