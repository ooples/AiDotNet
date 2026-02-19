using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FuSta (Full-frame Stabilization) video stabilization model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FuSta: Hybrid Approach for Full-frame Video Stabilization" (Liu et al., 2021)</item>
/// </list></para>
/// <para>
/// FuSta combines optical flow-based warping with outpainting to achieve full-frame stabilization
/// without cropping, using a two-stage pipeline: motion compensation + content completion.
/// </para>
/// </remarks>
public class FuStaOptions : ModelOptions
{
    /// <summary>
    /// Model variant controlling capacity and speed trade-off.
    /// </summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Number of base feature channels in the stabilization network.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Number of encoder-decoder levels in the outpainting branch.
    /// </summary>
    public int NumLevels { get; set; } = 4;

    /// <summary>
    /// Number of residual blocks in each encoder-decoder level.
    /// </summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>
    /// Number of attention heads for the content completion transformer.
    /// </summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Path to the ONNX model file for inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options for inference mode.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
