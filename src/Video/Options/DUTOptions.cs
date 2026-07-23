using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the DUT (Deep Unsupervised Trajectory) video stabilization model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "DUT: Learning Video Stabilization by Simply Watching Unstable Videos" (Xu et al., ICCV 2022)</item>
/// </list></para>
/// <para><b>For Beginners:</b> DUT options configure the Deep Unsupervised Trajectory video stabilizer.</para>
/// <para>
/// DUT learns stabilization in an unsupervised manner by watching unstable videos,
/// predicting per-pixel flow fields for warping without requiring paired stable/unstable data.
/// </para>
/// </remarks>
public class DUTOptions : ModelOptions
{
    #region Architecture

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public DUTOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DUTOptions(DUTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumPyramidLevels = other.NumPyramidLevels;
        NumResBlocks = other.NumResBlocks;
        TemporalWindowSize = other.TemporalWindowSize;
        TemporalLossWeight = other.TemporalLossWeight;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>
    /// Model variant controlling capacity and speed trade-off.
    /// </summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Number of base feature channels in the flow estimation network.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Number of pyramid levels for coarse-to-fine flow estimation.
    /// </summary>
    public int NumPyramidLevels { get; set; } = 4;

    /// <summary>
    /// Number of residual blocks per pyramid level.
    /// </summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>
    /// Temporal window size for multi-frame stabilization.
    /// </summary>
    public int TemporalWindowSize { get; set; } = 7;

    #endregion

    #region Model Loading

    /// <summary>Path to the ONNX model file for inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options for inference mode.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Weight for the temporal consistency loss during training.
    /// </summary>
    public double TemporalLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
