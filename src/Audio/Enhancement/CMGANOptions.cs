using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Configuration options for the CMGAN (Conformer-based Metric GAN) speech enhancement model.
/// </summary>
/// <remarks>
/// <para>
/// CMGAN (Cao et al., INTERSPEECH 2022) combines a conformer-based generator with a metric
/// discriminator for high-quality speech enhancement. It achieves PESQ 3.41 and STOI 0.97
/// on the VoiceBank-DEMAND dataset, outperforming previous GAN-based methods.
/// </para>
/// <para>
/// <b>For Beginners:</b> CMGAN uses a "competition" strategy to enhance speech:
/// <list type="number">
/// <item>A "generator" network tries to clean up noisy audio</item>
/// <item>A "discriminator" network judges how realistic the cleaned audio sounds</item>
/// <item>They train together: the generator gets better at fooling the discriminator</item>
/// <item>The result is very natural-sounding enhanced speech</item>
/// </list>
/// The "Conformer" part means it uses a mix of attention (looking at the whole signal)
/// and convolution (looking at local patterns), getting the best of both approaches.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "CMGAN: Conformer-based Metric GAN for Speech Enhancement" (Cao et al., INTERSPEECH 2022)</item>
/// <item>Repository: https://github.com/ruizhecao96/CMGAN</item>
/// </list>
/// </para>
/// </remarks>
public class CMGANOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int FftSize { get; set; } = 400;

    /// <summary>
    /// Gets or sets the hop length between frames.
    /// </summary>
    public int HopLength { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of frequency bins.
    /// </summary>
    public int NumFreqBins { get; set; } = 201;

    #endregion

    #region Generator Architecture

    /// <summary>
    /// Gets or sets the number of Conformer encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>Conformer layers combine self-attention and convolution for both global and local
    /// pattern recognition in the time-frequency domain.</para>
    /// </remarks>
    public int NumConformerLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the Conformer hidden dimension.
    /// </summary>
    public int ConformerDim { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of attention heads in Conformer layers.
    /// </summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the convolution kernel size in Conformer layers.
    /// </summary>
    public int ConformerKernelSize { get; set; } = 31;

    /// <summary>
    /// Gets or sets the number of U-Net encoder channels.
    /// </summary>
    /// <remarks>
    /// <para>CMGAN uses a U-Net structure where the encoder compresses the spectrogram and
    /// the decoder reconstructs it, with skip connections preserving detail.</para>
    /// </remarks>
    public int[] EncoderChannels { get; set; } = [16, 32, 64, 128, 256];

    /// <summary>
    /// Gets or sets the decoder kernel size.
    /// </summary>
    public int DecoderKernelSize { get; set; } = 5;

    #endregion

    #region Discriminator

    /// <summary>
    /// Gets or sets whether to use the metric discriminator during training.
    /// </summary>
    public bool UseDiscriminator { get; set; } = true;

    /// <summary>
    /// Gets or sets the discriminator learning rate.
    /// </summary>
    public double DiscriminatorLearningRate { get; set; } = 5e-4;

    #endregion

    #region Enhancement

    /// <summary>
    /// Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum).
    /// </summary>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets whether to enhance both magnitude and phase.
    /// </summary>
    public bool EnhancePhase { get; set; } = true;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the generator learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 1e-4;

    #endregion
}
