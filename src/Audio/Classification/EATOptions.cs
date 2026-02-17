using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the EAT (Efficient Audio Transformer) model.
/// </summary>
/// <remarks>
/// <para>
/// EAT (Chen et al., 2024) is an efficient self-supervised audio pre-training model that
/// achieves competitive performance with significantly less compute than previous methods.
/// It reaches 49.7% mAP on AudioSet-2M using only 10% of the pre-training data and compute
/// of BEATs.
/// </para>
/// <para>
/// <b>For Beginners:</b> EAT is designed to be more efficient while maintaining accuracy.
/// It uses a teacher-student framework where a smaller student model learns from a larger
/// teacher, making training much faster. Think of it as a student who learns efficiently
/// by watching an expert rather than figuring everything out alone.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer" (Chen et al., 2024)</item>
/// </list>
/// </para>
/// </remarks>
public class EATOptions : ModelOptions
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
    /// Gets or sets the hop length between FFT frames.
    /// </summary>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    public int NumMels { get; set; } = 128;

    /// <summary>
    /// Gets or sets the minimum frequency for mel filterbank.
    /// </summary>
    public int FMin { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum frequency for mel filterbank.
    /// </summary>
    public int FMax { get; set; } = 8000;

    #endregion

    #region Transformer Architecture

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>EAT uses 768 dimensions for the base model, matching ViT-Base.</para>
    /// </remarks>
    public int EmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of Transformer encoder layers.
    /// </summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the feed-forward network dimension.
    /// </summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the patch size.
    /// </summary>
    public int PatchSize { get; set; } = 16;

    /// <summary>
    /// Gets or sets the patch stride.
    /// </summary>
    public int PatchStride { get; set; } = 16;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the attention dropout rate.
    /// </summary>
    public double AttentionDropoutRate { get; set; } = 0.1;

    #endregion

    #region EAT-Specific: Teacher-Student

    /// <summary>
    /// Gets or sets the EMA (Exponential Moving Average) decay for teacher model updates.
    /// </summary>
    /// <remarks>
    /// <para>EAT uses a momentum teacher updated via EMA. The decay rate controls how quickly
    /// the teacher incorporates student improvements. Higher values = slower, more stable teacher.</para>
    /// <para><b>For Beginners:</b> The teacher model slowly follows the student's improvements,
    /// providing stable training targets.</para>
    /// </remarks>
    public double EmaDecay { get; set; } = 0.9998;

    /// <summary>
    /// Gets or sets the mask ratio for self-supervised pre-training.
    /// </summary>
    /// <remarks>
    /// <para>EAT masks 75% of patches during pre-training, similar to BEATs.</para>
    /// </remarks>
    public double MaskRatio { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the minimum span length for masking.
    /// </summary>
    public int MinMaskSpanLength { get; set; } = 2;

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold for event detection.
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
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>
    /// Gets or sets the number of warm-up steps.
    /// </summary>
    public int WarmUpSteps { get; set; } = 8000;

    /// <summary>
    /// Gets or sets the label smoothing factor.
    /// </summary>
    public double LabelSmoothing { get; set; } = 0.1;

    #endregion
}
