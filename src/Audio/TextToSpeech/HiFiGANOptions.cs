using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Configuration options for the HiFi-GAN neural vocoder.
/// </summary>
/// <remarks>
/// <para>
/// HiFi-GAN (Kong et al., 2020) is a GAN-based neural vocoder that converts mel-spectrograms
/// to high-fidelity waveforms in real-time. It uses multi-period and multi-scale discriminators
/// to model both periodic and non-periodic speech patterns. HiFi-GAN V1 achieves MOS 4.23
/// with 13.9x real-time on CPU, making it the standard vocoder for production TTS systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> A vocoder is the final step in a TTS pipeline. After a model like
/// StyleTTS 2 or Tacotron generates a mel-spectrogram (a visual representation of sound),
/// the vocoder converts it into actual audio you can hear. HiFi-GAN does this extremely
/// well and fast - it can generate audio faster than real-time, meaning it can create
/// 1 second of audio in less than 1 second of compute time.
///
/// Think of it like this: the TTS model writes sheet music, and the vocoder plays it.
/// </para>
/// </remarks>
public class HiFiGANOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the output audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>Gets or sets the number of mel-spectrogram frequency bins.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the hop length for spectrogram alignment.</summary>
    public int HopLength { get; set; } = 256;

    #endregion

    #region Generator Architecture

    /// <summary>Gets or sets the model variant ("V1", "V2", or "V3").</summary>
    /// <remarks>
    /// - V1: Best quality (MOS 4.23), 14M params, 13.9x real-time on CPU
    /// - V2: Balanced (MOS 4.17), 0.9M params, much faster
    /// - V3: Smallest (MOS 4.05), 1.5M params, fastest inference
    /// </remarks>
    public string Variant { get; set; } = "V1";

    /// <summary>Gets or sets the initial upsample channel count.</summary>
    public int UpsampleInitialChannel { get; set; } = 512;

    /// <summary>Gets or sets the upsample rates (product should equal hop length).</summary>
    /// <remarks>
    /// The product of all upsample rates should match the hop length.
    /// For example: [8, 8, 2, 2] with hop_length=256 means 8*8*2*2=256.
    /// </remarks>
    public int[] UpsampleRates { get; set; } = [8, 8, 2, 2];

    /// <summary>Gets or sets the upsample kernel sizes.</summary>
    public int[] UpsampleKernelSizes { get; set; } = [16, 16, 4, 4];

    /// <summary>Gets or sets the number of Multi-Receptive Field Fusion (MRF) blocks per layer.</summary>
    public int NumResBlocks { get; set; } = 3;

    /// <summary>Gets or sets the residual block kernel sizes.</summary>
    public int[][] ResBlockKernelSizes { get; set; } = [[3, 7, 11], [3, 7, 11], [3, 7, 11]];

    /// <summary>Gets or sets the residual block dilation sizes.</summary>
    public int[][] ResBlockDilationSizes { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate for the generator.</summary>
    public double GeneratorLearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the learning rate for the discriminator.</summary>
    public double DiscriminatorLearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the weight for feature matching loss.</summary>
    public double FeatureMatchingWeight { get; set; } = 2.0;

    /// <summary>Gets or sets the weight for mel-spectrogram reconstruction loss.</summary>
    public double MelReconstructionWeight { get; set; } = 45.0;

    #endregion
}
