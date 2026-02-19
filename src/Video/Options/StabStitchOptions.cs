using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the StabStitch video stabilization model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "StabStitch: Simultaneous Video Stabilization and Stitching" (2023)</item>
/// </list></para>
/// <para>
/// StabStitch jointly performs video stabilization and stitching, simultaneously removing
/// camera shake while producing a seamless panoramic output from moving cameras.
/// </para>
/// </remarks>
public class StabStitchOptions : ModelOptions
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
    /// Number of warp estimation branches (one per camera stream).
    /// </summary>
    public int NumWarpBranches { get; set; } = 2;

    /// <summary>
    /// Number of mesh grid rows for thin-plate-spline warping.
    /// </summary>
    public int MeshGridRows { get; set; } = 8;

    /// <summary>
    /// Number of mesh grid columns for thin-plate-spline warping.
    /// </summary>
    public int MeshGridCols { get; set; } = 8;

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
