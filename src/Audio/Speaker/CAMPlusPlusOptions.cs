using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the CAM++ (Context-Aware Masking Plus Plus) speaker model.
/// </summary>
/// <remarks>
/// <para>
/// CAM++ (Wang et al., 2023) is a fast and accurate speaker verification model that uses
/// context-aware masking with a densely connected time delay neural network (D-TDNN).
/// It processes variable-length utterances efficiently and achieves competitive EER results
/// while being significantly faster than Transformer-based approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b> CAM++ is a lightweight speaker recognition model optimized for speed.
/// It uses a clever "context-aware masking" technique that helps it focus on the most important
/// parts of speech for identifying who is speaking, while ignoring silence and noise. This makes
/// it both fast and accurate—ideal for real-time applications.
/// </para>
/// </remarks>
public class CAMPlusPlusOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 512;

    /// <summary>Gets or sets the hop length between frames.</summary>
    public int HopLength { get; set; } = 160;

    #endregion

    #region D-TDNN Architecture

    /// <summary>Gets or sets the initial channel dimension.</summary>
    public int InitialChannels { get; set; } = 512;

    /// <summary>Gets or sets the growth rate for dense connections.</summary>
    public int GrowthRate { get; set; } = 64;

    /// <summary>Gets or sets the number of D-TDNN blocks.</summary>
    public int NumBlocks { get; set; } = 6;

    /// <summary>Gets or sets the bottleneck dimension for D-TDNN blocks.</summary>
    public int BottleneckDim { get; set; } = 128;

    /// <summary>Gets or sets the context-aware masking dimension.</summary>
    public int MaskingDim { get; set; } = 256;

    #endregion

    #region Embedding

    /// <summary>Gets or sets the output speaker embedding dimension.</summary>
    public int EmbeddingDim { get; set; } = 192;

    /// <summary>Gets or sets the pooling dimension before embedding projection.</summary>
    public int PoolingDim { get; set; } = 1536;

    #endregion

    #region Verification

    /// <summary>Gets or sets the default cosine similarity threshold.</summary>
    public double DefaultThreshold { get; set; } = 0.6;

    /// <summary>Gets or sets the minimum audio duration in seconds.</summary>
    public double MinDurationSeconds { get; set; } = 0.3;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 5e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
