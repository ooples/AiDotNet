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
/// <para><b>For Beginners:</b> FlowLens options configure the flow-guided video inpainting model.</para>
/// <para>
/// FlowLens completes optical flow in masked regions first, then uses the completed flow
/// for high-quality temporal propagation followed by a refinement network, decoupling
/// motion estimation from pixel synthesis.
/// </para>
/// </remarks>
public class FlowLensOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public FlowLensOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FlowLensOptions(FlowLensOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumFlowIters = other.NumFlowIters;
        NumLevels = other.NumLevels;
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
