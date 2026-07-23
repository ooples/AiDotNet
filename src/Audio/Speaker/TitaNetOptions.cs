using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the TitaNet speaker verification and embedding model.
/// </summary>
/// <remarks>
/// <para>
/// TitaNet (Koluguri et al., ICASSP 2022) is NVIDIA's speaker embedding model based on
/// 1D depth-wise separable convolutions with Squeeze-Excitation and global context. TitaNet-L
/// achieves 0.68% EER on VoxCeleb1-O, outperforming ECAPA-TDNN.
/// </para>
/// <para>
/// <b>For Beginners:</b> TitaNet is NVIDIA's advanced voice fingerprinting model. It uses
/// efficient convolutions to process speech and creates a compact embedding that uniquely
/// identifies a speaker. It comes in three sizes: Small (S), Medium (M), and Large (L).
/// </para>
/// </remarks>
public class TitaNetOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the expected audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    public int NumMels { get; set; } = 80;

    /// <summary>
    /// Gets or sets the FFT window size in samples.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length between frames in samples.
    /// </summary>
    public int HopLength { get; set; } = 160;

    #endregion

    #region Encoder Architecture

    /// <summary>
    /// Gets or sets the model variant (S, M, or L).
    /// </summary>
    /// <remarks>
    /// <para>
    /// - "S" (Small): 6M params, 192-dim embedding
    /// - "M" (Medium): 13M params, 192-dim embedding
    /// - "L" (Large): 25M params, 192-dim embedding
    /// </para>
    /// </remarks>
    public string Variant { get; set; } = "L";

    /// <summary>
    /// Gets or sets the encoder hidden dimension.
    /// </summary>
    public int EncoderDim { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of encoder blocks (prolog + body + epilog).
    /// </summary>
    public int NumEncoderBlocks { get; set; } = 22;

    /// <summary>
    /// Gets or sets the depth-wise separable convolution kernel size.
    /// </summary>
    public int ConvKernelSize { get; set; } = 11;

    /// <summary>
    /// Gets or sets the SE (Squeeze-Excitation) reduction ratio.
    /// </summary>
    public int SEReductionRatio { get; set; } = 8;

    #endregion

    #region Embedding

    /// <summary>
    /// Gets or sets the output embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; set; } = 192;

    /// <summary>
    /// Gets or sets the attentive statistics pooling hidden dimension.
    /// </summary>
    public int AttentivePoolingDim { get; set; } = 128;

    #endregion

    #region Verification

    /// <summary>
    /// Gets or sets the default cosine similarity threshold for verification.
    /// </summary>
    public double DefaultThreshold { get; set; } = 0.6;

    /// <summary>
    /// Gets or sets the minimum audio duration in seconds for reliable embedding.
    /// </summary>
    public double MinDurationSeconds { get; set; } = 0.5;

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
    /// Gets or sets the learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 2e-5;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
