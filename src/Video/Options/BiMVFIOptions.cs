using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the BiMVFI bidirectional motion field model.
/// </summary>
/// <remarks>
/// <para>
/// BiMVFI (Seo et al., CVPR 2025) handles non-uniform motion with bidirectional fields:
/// - Bidirectional motion fields: estimates forward (0 to t) and backward (1 to t) motion
///   fields independently, each with its own confidence map, instead of a single symmetric flow
/// - Adaptive blending: per-pixel confidence weights learned from both motion fields determine
///   how to blend warped frames, handling occlusion regions where only one direction is valid
/// - Non-uniform motion modeling: dedicated occlusion reasoning module that detects regions with
///   non-uniform motion (e.g., independently moving objects) and applies motion-compensated
///   attention to those areas specifically
/// - Multi-scale architecture: 3-level feature pyramid with cross-scale feature propagation
///   for handling both small and large displacements
/// </para>
/// <para>
/// <b>For Beginners:</b> When objects move at different speeds or occlude each other, a single
/// motion estimate fails. BiMVFI solves this by estimating motion from both directions (past
/// and future) and letting each pixel choose which direction gives a better result. Where an
/// object appears in one direction but not the other (occlusion), it knows to trust only the
/// visible direction.
/// </para>
/// </remarks>
public class BiMVFIOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual blocks in the motion estimator.</summary>
    public int NumResBlocks { get; set; } = 8;

    /// <summary>Gets or sets the number of pyramid scales.</summary>
    public int NumScales { get; set; } = 3;

    /// <summary>Gets or sets whether to use occlusion-aware blending.</summary>
    public bool OcclusionAwareBlending { get; set; } = true;

    /// <summary>Gets or sets the confidence threshold for adaptive blending.</summary>
    /// <remarks>Below this threshold, the opposite direction's warp is preferred.</remarks>
    public double ConfidenceThreshold { get; set; } = 0.5;

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
