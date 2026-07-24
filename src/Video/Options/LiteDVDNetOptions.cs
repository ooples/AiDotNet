using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for LiteDVDNet lightweight deep video denoising.
/// </summary>
/// <remarks>
/// <para>
/// LiteDVDNet is a lightweight variant of DVDNet for efficient video denoising:
/// - Two-stage pipeline: first stage denoises each frame independently, second stage
///   fuses temporal information from the independently denoised frames
/// - Lightweight blocks: uses depthwise separable convolutions instead of standard
///   convolutions, reducing parameters by 8-10x while maintaining quality
/// - Non-blind support: accepts noise level sigma as input, allowing the network to
///   adapt its denoising strength to the actual noise level
/// - Efficient fusion: simple temporal fusion via 1x1 convolutions over stacked frames
///   rather than expensive optical flow or attention mechanisms
/// </para>
/// <para>
/// <b>For Beginners:</b> LiteDVDNet is a fast, lightweight video denoiser. It first cleans
/// each frame individually, then combines information from nearby frames to improve quality.
/// It's designed to run efficiently on devices with limited computing power.
/// </para>
/// </remarks>
public class LiteDVDNetOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public LiteDVDNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LiteDVDNetOptions(LiteDVDNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumBlocks = other.NumBlocks;
        TemporalWindowSize = other.TemporalWindowSize;
        ExpansionFactor = other.ExpansionFactor;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 48;

    /// <summary>Gets or sets the number of denoising blocks per stage.</summary>
    public int NumBlocks { get; set; } = 4;

    /// <summary>Gets or sets the temporal window size (number of input frames).</summary>
    public int TemporalWindowSize { get; set; } = 5;

    /// <summary>Gets or sets the depthwise separable expansion factor.</summary>
    public int ExpansionFactor { get; set; } = 2;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate. Default 1e-5 — this deep 10-layer conv encoder-decoder has no
    /// normalization and no residual skip (the 15-channel temporal input vs 3-channel output rules out a
    /// shape-safe global skip), so it overshoots at higher rates. Measured on Training_ShouldReduceLoss:
    /// 1e-3 exploded (0.28 -> 150), 1e-4 still rose (0.28 -> 1.77), 1e-5 trains stably (test passes). The
    /// model's [ResearchPaper] URL is a mis-citation to an unrelated paper, so there is no authoritative paper
    /// lr; this is a stable default, fully user-overridable via the constructor optimizer or this option.</summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
