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

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
