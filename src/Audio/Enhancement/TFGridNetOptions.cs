using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Configuration options for the TF-GridNet (Time-Frequency GridNet) speech enhancement model.
/// </summary>
/// <remarks>
/// <para>
/// TF-GridNet (Wang et al., ICASSP 2023) applies alternating attention along the time and frequency
/// axes in a grid pattern, achieving state-of-the-art performance. On the WSJ0-2mix benchmark it
/// reaches 23.4 dB SI-SNRi, and on DNS Challenge 2020 it achieves PESQ 3.41.
/// </para>
/// <para>
/// <b>For Beginners:</b> TF-GridNet processes audio like reading a grid:
/// <list type="number">
/// <item>Audio is represented as a time-frequency grid (spectrogram)</item>
/// <item>The network looks across time (left-right) to track how sounds change</item>
/// <item>Then it looks across frequency (up-down) to understand the harmonic structure</item>
/// <item>By alternating these views, it builds a complete understanding of the audio</item>
/// </list>
/// Imagine cleaning a dirty photo by first cleaning each row, then each column,
/// and repeating - that is how TF-GridNet cleans audio.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "TF-GridNet: Making Time-Frequency Domain Models Great Again" (Wang et al., ICASSP 2023)</item>
/// <item>Repository: https://github.com/espnet/espnet</item>
/// </list>
/// </para>
/// </remarks>
public class TFGridNetOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length between frames.
    /// </summary>
    public int HopLength { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of frequency bins.
    /// </summary>
    public int NumFreqBins { get; set; } = 257;

    #endregion

    #region Grid Architecture

    /// <summary>
    /// Gets or sets the number of grid blocks (each contains time + frequency attention).
    /// </summary>
    /// <remarks>
    /// <para>Each block applies attention along time, then along frequency. More blocks
    /// mean deeper processing but slower inference.</para>
    /// </remarks>
    public int NumBlocks { get; set; } = 6;

    /// <summary>
    /// Gets or sets the hidden dimension for the grid network.
    /// </summary>
    public int HiddenDim { get; set; } = 192;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the LSTM hidden size for intra-frame (frequency) processing.
    /// </summary>
    public int IntraFrameHiddenSize { get; set; } = 192;

    /// <summary>
    /// Gets or sets the LSTM hidden size for inter-frame (time) processing.
    /// </summary>
    public int InterFrameHiddenSize { get; set; } = 192;

    /// <summary>
    /// Gets or sets the embedding dimension for each T-F bin.
    /// </summary>
    public int EmbeddingDim { get; set; } = 48;

    #endregion

    #region Enhancement

    /// <summary>
    /// Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum).
    /// </summary>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the number of sources (1 for enhancement, 2+ for separation).
    /// </summary>
    public int NumSources { get; set; } = 1;

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
    /// Gets or sets the initial learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 1e-5;

    #endregion
}
