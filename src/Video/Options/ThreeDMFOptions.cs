using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the 3DMF (3D Motion Field) video stabilization model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "3D Video Stabilization with Depth Estimation by CNN-based Optimization" (Lee &amp; Lee, CVPR 2021)</item>
/// </list></para>
/// <para><b>For Beginners:</b> 3DMF options configure the 3D motion field video stabilizer.</para>
/// <para>
/// 3DMF estimates depth and 3D camera motion to perform stabilization in 3D space,
/// better handling parallax and depth-dependent motion than 2D methods.
/// </para>
/// </remarks>
public class ThreeDMFOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public ThreeDMFOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ThreeDMFOptions(ThreeDMFOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDepthLayers = other.NumDepthLayers;
        NumMotionIters = other.NumMotionIters;
        NumResBlocks = other.NumResBlocks;
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
    /// Number of base feature channels.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Number of depth estimation layers in the depth prediction branch.
    /// </summary>
    public int NumDepthLayers { get; set; } = 4;

    /// <summary>
    /// Number of 3D motion estimation refinement iterations.
    /// </summary>
    public int NumMotionIters { get; set; } = 3;

    /// <summary>
    /// Number of residual blocks in the feature backbone.
    /// </summary>
    public int NumResBlocks { get; set; } = 2;

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
