using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the SoundStream neural audio codec model.
/// </summary>
/// <remarks>
/// <para>
/// SoundStream (Zeghidour et al., 2021, Google) is a neural audio codec that compresses
/// audio at 3-18 kbps using a fully convolutional encoder-decoder with residual vector
/// quantization. It pioneered the RVQ approach later adopted by EnCodec, and powers
/// Google's AudioLM and MusicLM systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> SoundStream is Google's version of a neural audio compressor.
/// It squeezes audio into tiny tokens and reconstructs it back. The key innovation is
/// "residual vector quantization" (RVQ) - it uses multiple codebooks, each one refining
/// the previous approximation, like painting with increasingly fine brushstrokes.
/// </para>
/// </remarks>
public class SoundStreamOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>
    /// Gets or sets the audio sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>
    /// Gets or sets the number of audio channels.
    /// </summary>
    public int Channels { get; set; } = 1;

    #endregion

    #region Encoder Architecture

    /// <summary>
    /// Gets or sets the encoder channel dimensions.
    /// </summary>
    public int[] EncoderChannels { get; set; } = [32, 64, 128, 256];

    /// <summary>
    /// Gets or sets the temporal downsampling ratios.
    /// </summary>
    public int[] DownsampleRatios { get; set; } = [2, 4, 5, 8];

    /// <summary>
    /// Gets or sets the encoder output dimension (embedding dimension).
    /// </summary>
    public int EncoderDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of residual blocks per stage.
    /// </summary>
    public int NumResBlocks { get; set; } = 3;

    #endregion

    #region Quantization

    /// <summary>
    /// Gets or sets the number of residual vector quantizers.
    /// </summary>
    public int NumQuantizers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the codebook size per quantizer.
    /// </summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the codebook dimension.
    /// </summary>
    public int CodebookDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the target bitrate in kbps.
    /// </summary>
    public double TargetBitrateKbps { get; set; } = 6.0;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
