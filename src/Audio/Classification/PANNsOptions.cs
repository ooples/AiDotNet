using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the PANNs (Pre-trained Audio Neural Networks) model.
/// </summary>
/// <remarks>
/// <para>
/// PANNs (Kong et al., IEEE/ACM TASLP 2020) provides a comprehensive set of pre-trained
/// CNN-based audio classification models. The flagship CNN14 model achieves 43.1% mAP on
/// AudioSet-2M and has become one of the most widely-used audio feature extractors.
/// </para>
/// <para>
/// <b>For Beginners:</b> PANNs is a family of CNN-based audio classifiers. Unlike Transformer
/// models (AST, BEATs), PANNs uses convolutional neural networks (CNNs) - the same technology
/// used for image recognition. CNNs are good at detecting local patterns (like specific
/// frequency shapes) and combining them into higher-level understanding (like "this is a dog bark").
/// PANNs is fast, well-tested, and widely used as a feature extractor for other audio tasks.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition" (Kong et al., 2020)</item>
/// <item>Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn</item>
/// </list>
/// </para>
/// </remarks>
public class PANNsOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// <para>PANNs uses 32 kHz audio for better frequency coverage.</para>
    /// </remarks>
    public int SampleRate { get; set; } = 32000;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 320;

    /// <summary>
    /// Gets or sets the number of mel bands.
    /// </summary>
    /// <remarks>
    /// <para>PANNs CNN14 uses 64 mel bands.</para>
    /// </remarks>
    public int NumMels { get; set; } = 64;

    /// <summary>
    /// Gets or sets the minimum frequency.
    /// </summary>
    public int FMin { get; set; } = 50;

    /// <summary>
    /// Gets or sets the maximum frequency.
    /// </summary>
    public int FMax { get; set; } = 14000;

    #endregion

    #region CNN Architecture

    /// <summary>
    /// Gets or sets the number of CNN blocks.
    /// </summary>
    /// <remarks>
    /// <para>CNN14 has 6 convolutional blocks with increasing channel counts
    /// (64, 128, 256, 512, 1024, 2048).</para>
    /// <para><b>For Beginners:</b> Each block learns more complex audio patterns.
    /// Early blocks detect simple patterns (pitch, volume), later blocks detect
    /// complex patterns (instruments, speech characteristics).</para>
    /// </remarks>
    public int NumBlocks { get; set; } = 6;

    /// <summary>
    /// Gets or sets the base channel count.
    /// </summary>
    /// <remarks>
    /// <para>The first block starts with this many channels, doubling at each subsequent block.</para>
    /// </remarks>
    public int BaseChannels { get; set; } = 64;

    /// <summary>
    /// Gets or sets the embedding dimension for the classification head.
    /// </summary>
    /// <remarks>
    /// <para>CNN14 uses 2048-dimensional embeddings before the final classification layer.</para>
    /// </remarks>
    public int EmbeddingDim { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.2;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold.
    /// </summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the window size in seconds.
    /// </summary>
    public double WindowSize { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the window overlap ratio.
    /// </summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets custom event labels.
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
