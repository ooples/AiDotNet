using AiDotNet.Onnx;

namespace AiDotNet.Audio.AudioLDM;

/// <summary>
/// Configuration options for AudioLDM text-to-audio generation.
/// </summary>
/// <remarks>
/// <para>
/// AudioLDM is a latent diffusion model for text-to-audio generation. It operates
/// in a compressed latent space learned by a VAE, making generation efficient while
/// maintaining high audio quality.
/// </para>
/// <para><b>For Beginners:</b> AudioLDM generates realistic audio from descriptions:
///
/// Example prompts:
/// - "A dog barking followed by children laughing"
/// - "Rain falling on a tin roof with distant thunder"
/// - "Footsteps on gravel approaching and stopping"
/// - "Piano music in a concert hall with audience applause"
///
/// Tips for good prompts:
/// - Be specific about the sound source and environment
/// - Include temporal information (before, after, while)
/// - Mention acoustic properties (loud, soft, distant, echoing)
/// </para>
/// </remarks>
public class AudioLDMOptions : AiDotNet.Models.Options.ModelOptions
{
    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <remarks>
    /// Different sizes trade off quality vs speed.
    /// Default is Base which balances both well.
    /// </remarks>
    public AudioLDMModelSize ModelSize { get; set; } = AudioLDMModelSize.Base;

    /// <summary>
    /// Gets or sets the output sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// AudioLDM uses 16kHz by default for speech/environmental sounds.
    /// Can be set to 48kHz for higher quality music generation.
    /// </remarks>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the default duration of generated audio in seconds.
    /// </summary>
    public double DurationSeconds { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets the maximum duration in seconds.
    /// </summary>
    /// <remarks>
    /// AudioLDM can generate up to 30 seconds of audio.
    /// Longer durations require more memory and compute time.
    /// </remarks>
    public double MaxDurationSeconds { get; set; } = 30.0;

    /// <summary>
    /// Gets or sets the number of diffusion steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More steps = higher quality but slower generation:
    /// - 25 steps: Fast, lower quality
    /// - 50 steps: Good balance (default)
    /// - 100+ steps: Best quality, slow
    /// </para>
    /// </remarks>
    public int NumInferenceSteps { get; set; } = 50;

    /// <summary>
    /// Gets or sets the classifier-free guidance scale.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls how closely the model follows the text prompt:
    /// - Low (1.0-2.0): More variation, less prompt adherence
    /// - Default (2.5): Good balance
    /// - High (4.0-7.0): Stricter prompt following
    /// </para>
    /// </remarks>
    public double GuidanceScale { get; set; } = 2.5;

    /// <summary>
    /// Gets or sets the number of mel spectrogram bins.
    /// </summary>
    /// <remarks>
    /// Standard AudioLDM uses 64 mel bins.
    /// Higher values capture more spectral detail.
    /// </remarks>
    public int NumMelBins { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hop length for spectrogram computation.
    /// </summary>
    /// <remarks>
    /// Controls the time resolution of the spectrogram.
    /// Smaller values = higher time resolution but more compute.
    /// </remarks>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the VAE latent dimension.
    /// </summary>
    /// <remarks>
    /// The dimension of the compressed audio representation.
    /// Default of 8 matches standard AudioLDM architecture.
    /// </remarks>
    public int LatentDimension { get; set; } = 8;

    /// <summary>
    /// Gets or sets the latent downsampling factor.
    /// </summary>
    /// <remarks>
    /// How much the VAE compresses the spectrogram spatially.
    /// Default of 4 provides good compression/quality trade-off.
    /// </remarks>
    public int LatentDownsampleFactor { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to generate stereo audio.
    /// </summary>
    /// <remarks>
    /// When true, generates two-channel stereo output.
    /// Mono output is duplicated to stereo for compatibility.
    /// </remarks>
    public bool Stereo { get; set; } = false;

    /// <summary>
    /// Gets or sets the path to the CLAP text encoder ONNX model.
    /// </summary>
    public string? ClapEncoderPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the VAE ONNX model.
    /// </summary>
    public string? VaePath { get; set; }

    /// <summary>
    /// Gets or sets the path to the U-Net denoiser ONNX model.
    /// </summary>
    public string? UNetPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the HiFi-GAN vocoder ONNX model.
    /// </summary>
    public string? VocoderPath { get; set; }

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
    public int MaxTextLength { get; set; } = 77;

    /// <summary>
    /// Gets or sets the CLAP embedding dimension.
    /// </summary>
    public int ClapEmbeddingDim { get; set; } = 512;
}
