using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Configuration options for the BigVGAN neural vocoder.
/// </summary>
/// <remarks>
/// <para>
/// BigVGAN (Lee et al., 2023, NVIDIA) is a universal neural vocoder that generates
/// high-fidelity audio for both speech and non-speech (music, environmental sounds).
/// It uses anti-aliased multi-periodicity composition (AMP) with Snake activation functions
/// for improved periodic signal modeling. BigVGAN v2 achieves state-of-the-art quality
/// (PESQ 4.2, UTMOS 4.2) with up to 44.1 kHz output at 2x real-time on GPU.
/// </para>
/// <para>
/// <b>For Beginners:</b> BigVGAN is like an upgraded HiFi-GAN that can handle not just speech,
/// but also music and sound effects. While HiFi-GAN sometimes struggles with singing voices
/// or musical instruments, BigVGAN handles them all beautifully. Key improvements:
///
/// - Handles all audio types (speech, music, environmental sounds)
/// - Better at periodic signals (music notes, sine waves)
/// - Supports high sample rates (up to 44.1 kHz for music quality)
/// - More robust to out-of-distribution inputs
/// </para>
/// </remarks>
public class BigVGANOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the output audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>Gets or sets the number of mel-spectrogram frequency bins.</summary>
    public int NumMels { get; set; } = 100;

    /// <summary>Gets or sets the hop length for spectrogram alignment.</summary>
    public int HopLength { get; set; } = 256;

    /// <summary>Gets or sets the FFT size for spectrogram computation.</summary>
    public int FFTSize { get; set; } = 1024;

    #endregion

    #region Generator Architecture

    /// <summary>Gets or sets the model variant ("base", "v2", "v2_44khz").</summary>
    /// <remarks>
    /// - "base": 24 kHz, 112M params, universal vocoder
    /// - "v2": 24 kHz, 112M params, improved anti-aliasing
    /// - "v2_44khz": 44.1 kHz, for music/high-fidelity applications
    /// </remarks>
    public string Variant { get; set; } = "v2";

    /// <summary>Gets or sets the initial upsample channel count.</summary>
    public int UpsampleInitialChannel { get; set; } = 1536;

    /// <summary>Gets or sets the upsample rates.</summary>
    public int[] UpsampleRates { get; set; } = [4, 4, 2, 2, 2, 2];

    /// <summary>Gets or sets the upsample kernel sizes.</summary>
    public int[] UpsampleKernelSizes { get; set; } = [8, 8, 4, 4, 4, 4];

    /// <summary>Gets or sets the number of residual blocks per upsampling layer.</summary>
    public int NumResBlocks { get; set; } = 3;

    /// <summary>Gets or sets the residual block kernel sizes.</summary>
    public int[] ResBlockKernelSizes { get; set; } = [3, 7, 11];

    /// <summary>Gets or sets the residual block dilation sizes.</summary>
    public int[][] ResBlockDilationSizes { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    #endregion

    #region Snake Activation

    /// <summary>Gets or sets whether to use Snake activation (periodic activation function).</summary>
    public bool UseSnakeActivation { get; set; } = true;

    /// <summary>Gets or sets the initial alpha parameter for Snake activation.</summary>
    public double SnakeAlpha { get; set; } = 1.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the generator learning rate.</summary>
    public double GeneratorLearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the discriminator learning rate.</summary>
    public double DiscriminatorLearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
