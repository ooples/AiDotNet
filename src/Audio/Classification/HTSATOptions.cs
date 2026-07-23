using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the HTS-AT (Hierarchical Token-Semantic Audio Transformer) model.
/// </summary>
/// <remarks>
/// <para>
/// HTS-AT (Chen et al., ICASSP 2022) is a hierarchical Transformer architecture for audio
/// classification that uses Swin Transformer blocks with token-semantic modules to efficiently
/// process audio spectrograms. It achieves 47.1% mAP on AudioSet-2M with only 30M parameters.
/// </para>
/// <para>
/// <b>For Beginners:</b> HTS-AT is an efficient audio classifier that processes spectrograms
/// hierarchically (like zooming from overview to detail). It uses a technique called "window
/// attention" that looks at local regions first, then gradually combines them. This makes it
/// faster and more memory-efficient than models like AST that look at everything at once.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "HTS-AT: A Hierarchical Token-Semantic Audio Transformer" (Chen et al., ICASSP 2022)</item>
/// <item>Repository: https://github.com/RetroCirce/HTS-Audio-Transformer</item>
/// </list>
/// </para>
/// </remarks>
public class HTSATOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 32000;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the hop length between FFT frames.
    /// </summary>
    public int HopLength { get; set; } = 320;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    /// <remarks>
    /// <para>HTS-AT uses 64 mel bands by default.</para>
    /// </remarks>
    public int NumMels { get; set; } = 64;

    /// <summary>
    /// Gets or sets the minimum frequency for mel filterbank.
    /// </summary>
    public int FMin { get; set; } = 50;

    /// <summary>
    /// Gets or sets the maximum frequency for mel filterbank.
    /// </summary>
    public int FMax { get; set; } = 14000;

    #endregion

    #region Swin Transformer Architecture

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>HTS-AT uses 96 as the base embedding dimension, which doubles at each stage
    /// (96 -> 192 -> 384 -> 768).</para>
    /// <para><b>For Beginners:</b> The base feature size that grows as the model processes
    /// deeper levels of the hierarchy.</para>
    /// </remarks>
    public int EmbeddingDim { get; set; } = 96;

    /// <summary>
    /// Gets or sets the number of Swin Transformer layers per stage.
    /// </summary>
    /// <remarks>
    /// <para>HTS-AT uses [2, 2, 6, 2] layers in its four stages. The third stage has
    /// more layers as it operates at the most informative resolution.</para>
    /// </remarks>
    public int[] NumLayersPerStage { get; set; } = [2, 2, 6, 2];

    /// <summary>
    /// Gets or sets the number of attention heads per stage.
    /// </summary>
    public int[] NumHeadsPerStage { get; set; } = [4, 8, 16, 32];

    /// <summary>
    /// Gets or sets the window size for local attention.
    /// </summary>
    /// <remarks>
    /// <para>Swin Transformer uses local window attention instead of global attention.
    /// Window size 8 means attention is computed within 8x8 token groups.</para>
    /// <para><b>For Beginners:</b> Instead of each patch looking at all other patches
    /// (which is slow), HTS-AT divides patches into small windows and computes attention
    /// within each window. This is much faster.</para>
    /// </remarks>
    public int WindowSize { get; set; } = 8;

    /// <summary>
    /// Gets or sets the patch size for initial embedding.
    /// </summary>
    public int PatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the feed-forward expansion ratio.
    /// </summary>
    public double FeedForwardRatio { get; set; } = 4.0;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the attention dropout rate.
    /// </summary>
    public double AttentionDropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the drop path rate for stochastic depth.
    /// </summary>
    public double DropPathRate { get; set; } = 0.1;

    #endregion

    #region Token-Semantic Module

    /// <summary>
    /// Gets or sets whether to use the token-semantic module.
    /// </summary>
    /// <remarks>
    /// <para>The token-semantic module groups tokens by semantic meaning and aggregates them
    /// for classification. This improves accuracy by capturing global context efficiently.</para>
    /// <para><b>For Beginners:</b> This module groups similar audio patterns together before
    /// making the final classification, improving the model's understanding.</para>
    /// </remarks>
    public bool UseTokenSemanticModule { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of semantic groups.
    /// </summary>
    public int NumSemanticGroups { get; set; } = 4;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold for event detection.
    /// </summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the window size in seconds for event detection.
    /// </summary>
    public double DetectionWindowSize { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the window overlap ratio (0-1).
    /// </summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets custom event labels. If null, uses AudioSet-527 labels.
    /// </summary>
    public string[]? CustomLabels { get; set; }

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
    /// Gets or sets the warm-up steps.
    /// </summary>
    public int WarmUpSteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the label smoothing factor.
    /// </summary>
    public double LabelSmoothing { get; set; } = 0.1;

    #endregion
}
