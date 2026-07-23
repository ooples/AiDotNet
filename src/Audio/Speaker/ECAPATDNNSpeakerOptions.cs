using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the ECAPA-TDNN speaker verification and embedding model.
/// </summary>
/// <remarks>
/// <para>
/// ECAPA-TDNN (Desplanques et al., Interspeech 2020) is a state-of-the-art speaker embedding
/// model that extends x-vector architecture with Squeeze-Excitation blocks, multi-layer feature
/// aggregation, and channel- and context-dependent statistics pooling. Achieves 0.87% EER on
/// VoxCeleb1 test set.
/// </para>
/// <para>
/// <b>For Beginners:</b> ECAPA-TDNN creates a "voiceprint" for any speaker. It processes
/// audio through special layers that focus on the most important voice characteristics.
/// The result is a compact vector (embedding) that uniquely identifies a speaker's voice.
/// </para>
/// </remarks>
public class ECAPATDNNSpeakerOptions : ModelOptions
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

    #region TDNN Architecture

    /// <summary>
    /// Gets or sets the channel dimensions for each TDNN block.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is [512, 512, 512, 512, 1536] matching the original ECAPA-TDNN paper.
    /// Each value specifies the output channels of a Res2Net-style TDNN block.
    /// </para>
    /// </remarks>
    public int[] Channels { get; set; } = [512, 512, 512, 512, 1536];

    /// <summary>
    /// Gets or sets the TDNN kernel sizes (dilation factors).
    /// </summary>
    public int[] KernelSizes { get; set; } = [5, 3, 3, 3, 1];

    /// <summary>
    /// Gets or sets the dilation factors for each TDNN block.
    /// </summary>
    public int[] Dilations { get; set; } = [1, 2, 3, 4, 1];

    /// <summary>
    /// Gets or sets the Res2Net scale factor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the number of parallel branches in Res2Net blocks.
    /// Higher values increase model capacity at the cost of computation.
    /// </para>
    /// </remarks>
    public int Res2NetScale { get; set; } = 8;

    /// <summary>
    /// Gets or sets the SE (Squeeze-Excitation) bottleneck dimension.
    /// </summary>
    public int SEBottleneckDim { get; set; } = 128;

    #endregion

    #region Embedding

    /// <summary>
    /// Gets or sets the output embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Standard values are 192 or 256. The embedding vector uniquely represents
    /// a speaker's voice characteristics.
    /// </para>
    /// </remarks>
    public int EmbeddingDim { get; set; } = 192;

    /// <summary>
    /// Gets or sets the pooling dimension before the final embedding projection.
    /// </summary>
    public int PoolingDim { get; set; } = 1536;

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
