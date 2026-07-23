using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Configuration options for the Descript Audio Codec (DAC) model.
/// </summary>
/// <remarks>
/// <para>
/// DAC (Kumar et al., 2024, Descript) is a high-fidelity universal neural audio codec
/// that compresses audio to 8 kbps while maintaining near-lossless quality. It uses
/// residual vector quantization (RVQ) with improved codebook usage, periodic activation
/// functions (Snake), and multi-scale STFT discriminators. DAC handles speech, music,
/// and environmental sounds at 16/24/44.1 kHz.
/// </para>
/// <para>
/// <b>For Beginners:</b> DAC is like a super-efficient audio compressor. While MP3 typically
/// uses 128-320 kbps, DAC achieves similar quality at just 8 kbps (16-40x smaller files).
/// It works by:
///
/// 1. <b>Encoding</b>: Converting audio into compact numerical codes (tokens)
/// 2. <b>Quantizing</b>: Discretizing the codes into a small set of entries
/// 3. <b>Decoding</b>: Reconstructing audio from the tokens
///
/// Unlike EnCodec, DAC uses improved quantization techniques and periodic activations
/// for better reconstruction of music and complex audio.
/// </para>
/// </remarks>
public class DACOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Gets or sets the number of audio channels.</summary>
    public int NumChannels { get; set; } = 1;

    #endregion

    #region Codec Architecture

    /// <summary>Gets or sets the model variant ("16khz", "24khz", "44khz").</summary>
    public string Variant { get; set; } = "44khz";

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int EncoderDim { get; set; } = 64;

    /// <summary>Gets or sets the number of encoder channels per downsampling stage.</summary>
    public int[] EncoderChannels { get; set; } = [64, 128, 256, 512];

    /// <summary>Gets or sets the number of codebooks (residual quantization levels).</summary>
    public int NumCodebooks { get; set; } = 9;

    /// <summary>Gets or sets the codebook size (entries per codebook).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the codebook dimension.</summary>
    public int CodebookDim { get; set; } = 8;

    /// <summary>Gets or sets the frame rate of encoded tokens (tokens per second).</summary>
    public int TokenFrameRate { get; set; } = 86;

    /// <summary>Gets or sets the target bitrate in kbps.</summary>
    public double TargetBitrate { get; set; } = 8.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the commitment loss weight for codebook training.</summary>
    public double CommitmentLossWeight { get; set; } = 0.25;

    #endregion
}
