using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the PWStableNet (Pixel-Wise Stable Net) video stabilization model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "PWStableNet: Learning Pixel-Wise Warping Maps for Video Stabilization" (Zhao et al., IEEE TIP 2020)</item>
/// </list></para>
/// <para>
/// PWStableNet predicts per-pixel warping maps (not global homographies) for more flexible
/// stabilization that handles parallax and rolling shutter distortion.
/// </para>
/// </remarks>
public class PWStableNetOptions : ModelOptions
{
    /// <summary>
    /// Model variant controlling capacity and speed trade-off.
    /// </summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Number of base feature channels.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Number of coarse-to-fine refinement iterations for the warp field.
    /// </summary>
    public int NumRefinementIters { get; set; } = 3;

    /// <summary>
    /// Grid size for the spatial transformer network that produces warp fields.
    /// </summary>
    public int GridSize { get; set; } = 8;

    /// <summary>
    /// Number of residual blocks in the feature extraction backbone.
    /// </summary>
    public int NumResBlocks { get; set; } = 4;

    /// <summary>
    /// Learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Path to the ONNX model file for inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options for inference mode.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
