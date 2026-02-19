using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for PerVFI perception-oriented video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// PerVFI (2024) uses perceptual quality optimization for frame interpolation:
/// - Perceptual loss hierarchy: instead of optimizing only pixel-level L1/L2 loss, PerVFI uses
///   a multi-scale perceptual loss computed from features of a pre-trained VGG/LPIPS network,
///   prioritizing visual quality over pixel-exact reconstruction
/// - Asymmetric distortion: applies different loss weights to different frequency bands,
///   penalizing low-frequency (structural) errors more heavily than high-frequency (texture)
///   errors, matching human visual perception priorities
/// - Perception-guided refinement: a refinement network that takes the initial interpolation
///   result plus a perceptual error map and iteratively improves visual quality in regions
///   where the perceptual loss is highest
/// - Motion-aware perceptual attention: uses estimated motion magnitude to weight the perceptual
///   loss, applying stronger perceptual constraints in high-motion regions where artifacts are
///   most visible
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods try to match pixels exactly, but
/// human eyes care more about visual quality than pixel accuracy. PerVFI optimizes for what
/// "looks good" rather than what's mathematically closest, producing results that appear
/// sharper and more natural even if pixel differences are slightly larger.
/// </para>
/// </remarks>
public class PerVFIOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of perceptual refinement iterations.</summary>
    public int NumRefinementIters { get; set; } = 3;

    /// <summary>Gets or sets the number of flow estimation scales.</summary>
    public int NumFlowScales { get; set; } = 4;

    /// <summary>Gets or sets the number of residual blocks in the refinement network.</summary>
    public int NumResBlocks { get; set; } = 4;

    /// <summary>Gets or sets the perceptual loss weight relative to reconstruction loss.</summary>
    public double PerceptualWeight { get; set; } = 0.1;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
