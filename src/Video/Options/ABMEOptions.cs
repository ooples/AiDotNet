using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the ABME asymmetric bilateral motion estimation model.
/// </summary>
/// <remarks>
/// <para>
/// ABME (Park et al., ICCV 2021) uses asymmetric bilateral motion estimation:
/// - Bilateral motion estimation: estimates motion from the target time to both input frames
///   simultaneously (t to 0 and t to 1), rather than from input frames toward the target
/// - Asymmetric motion model: the two bilateral motion fields are NOT assumed symmetric;
///   each has its own magnitude and direction, correctly handling non-linear motion paths
///   (e.g., accelerating objects, curved trajectories)
/// - Iterative refinement with asymmetric updates: a GRU-based module iteratively refines
///   both bilateral flows, with separate update heads that can correct each flow independently
/// - Context-aware synthesis: the final frame is synthesized by combining bilaterally warped
///   features with a learned blending mask that accounts for occlusion and motion boundaries
/// </para>
/// <para>
/// <b>For Beginners:</b> Most methods assume motion is symmetric (if something moves right
/// from frame 0, it moves left by the same amount from frame 1). But real motion isn't
/// symmetric -- a ball speeding up moves more in the second half. ABME estimates motion
/// independently in both directions, so it handles acceleration, deceleration, and curved
/// paths much better than symmetric methods.
/// </para>
/// </remarks>
public class ABMEOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual blocks in the encoder.</summary>
    public int NumResBlocks { get; set; } = 6;

    /// <summary>Gets or sets the number of GRU refinement iterations for bilateral flow.</summary>
    /// <remarks>More iterations improve accuracy. Paper uses 8 iterations.</remarks>
    public int NumRefinementIters { get; set; } = 8;

    /// <summary>Gets or sets the number of pyramid levels for coarse-to-fine estimation.</summary>
    public int NumPyramidLevels { get; set; } = 3;

    /// <summary>Gets or sets whether to use asymmetric motion modeling.</summary>
    /// <remarks>When false, bilateral flows are constrained to be symmetric.</remarks>
    public bool AsymmetricMotion { get; set; } = true;

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
