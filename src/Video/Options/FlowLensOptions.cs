using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FlowLens video inpainting model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FlowLens: Seeing Beyond the FoV via Optical Flow Completion" (Xu et al., ECCV 2022)</item>
/// </list></para>
/// <para>
/// FlowLens completes optical flow in masked regions first, then uses the completed flow
/// for high-quality temporal propagation followed by a refinement network, decoupling
/// motion estimation from pixel synthesis.
/// </para>
/// </remarks>
public class FlowLensOptions : ModelOptions
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
    /// Number of flow completion iterations for refining estimated flow in masked regions.
    /// </summary>
    public int NumFlowIters { get; set; } = 3;

    /// <summary>
    /// Number of encoder-decoder levels in the refinement network.
    /// </summary>
    public int NumLevels { get; set; } = 4;

    /// <summary>
    /// Number of residual blocks in the refinement branch.
    /// </summary>
    public int NumResBlocks { get; set; } = 2;

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
