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
/// The defaults are the released model's (3D-Speaker <c>CAMPPlus</c>, the <c>campplus.onnx</c> CosyVoice
/// ships): 80 Kaldi filterbank bands at 16 kHz, a 128-channel TDNN layer, D-TDNN blocks of 12, 24 and 16
/// layers growing 32 channels each through a 4×32 bottleneck, and a 192-wide embedding.
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

    /// <summary>Gets or sets the number of Kaldi filterbank bands (<c>feat_dim</c>); a multiple of 8.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the filterbank frame length in milliseconds (Kaldi's default).</summary>
    public double FrameLengthMilliseconds { get; set; } = 25.0;

    /// <summary>Gets or sets the filterbank frame shift in milliseconds (Kaldi's default).</summary>
    public double FrameShiftMilliseconds { get; set; } = 10.0;

    #endregion

    #region D-TDNN Architecture

    /// <summary>Gets or sets the TDNN layer's output channels (<c>init_channels</c>).</summary>
    public int InitialChannels { get; set; } = 128;

    /// <summary>Gets or sets the channels each D-TDNN layer adds (<c>growth_rate</c>).</summary>
    public int GrowthRate { get; set; } = 32;

    /// <summary>Gets or sets the D-TDNN bottleneck as a multiple of the growth rate (<c>bn_size</c>).</summary>
    public int BottleneckMultiplier { get; set; } = 4;

    #endregion

    #region Embedding

    /// <summary>Gets or sets the output speaker embedding dimension (<c>embedding_size</c>).</summary>
    public int EmbeddingDim { get; set; } = 192;

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

    /// <summary>Gets or sets the learning rate of the AdamW fallback, used only when no paper optimizer is built.</summary>
    public double LearningRate { get; set; } = 1e-3;

    #endregion
}
