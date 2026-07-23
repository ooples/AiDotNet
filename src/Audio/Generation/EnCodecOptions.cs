using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// Configuration options for the EnCodec neural audio codec model.
/// </summary>
/// <remarks>
/// <para>
/// EnCodec (Defossez et al., 2022, Meta) is a neural audio codec that compresses audio to
/// 1.5-24 kbps using residual vector quantization (RVQ). It uses an encoder-decoder architecture
/// with a multi-scale discriminator for adversarial training, achieving near-transparent quality
/// at 6 kbps. EnCodec tokens are widely used as input for audio language models.
/// </para>
/// <para>
/// <b>For Beginners:</b> EnCodec is like an AI-powered MP3. It compresses audio into tiny
/// tokens (numbers) and decompresses them back. At 6 kbps it sounds almost as good as the
/// original, while MP3 needs 128 kbps for similar quality. The compressed tokens are also
/// used by AI models that generate speech and music.
/// </para>
/// </remarks>
public class EnCodecOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>
    /// Gets or sets the audio sample rate (24 kHz or 48 kHz).
    /// </summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>
    /// Gets or sets the number of audio channels (1=mono, 2=stereo).
    /// </summary>
    public int Channels { get; set; } = 1;

    #endregion

    #region Encoder Architecture

    /// <summary>
    /// Gets or sets the encoder channel dimensions at each downsampling stage.
    /// </summary>
    public int[] EncoderChannels { get; set; } = [32, 64, 128, 256, 512];

    /// <summary>
    /// Gets or sets the downsampling ratios at each stage.
    /// </summary>
    public int[] DownsampleRatios { get; set; } = [8, 5, 4, 2];

    /// <summary>
    /// Gets or sets the encoder output dimension.
    /// </summary>
    public int EncoderDim { get; set; } = 128;

    #endregion

    #region Quantization

    /// <summary>
    /// Gets or sets the number of residual vector quantizers.
    /// </summary>
    public int NumQuantizers { get; set; } = 8;

    /// <summary>
    /// Gets or sets the codebook size per quantizer.
    /// </summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the codebook dimension.
    /// </summary>
    public int CodebookDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the target bandwidth in kbps (determines number of active quantizers).
    /// </summary>
    public double TargetBandwidthKbps { get; set; } = 6.0;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to the ONNX encoder model.
    /// </summary>
    public string? EncoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the ONNX decoder model.
    /// </summary>
    public string? DecoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets the combined ONNX model path.
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
