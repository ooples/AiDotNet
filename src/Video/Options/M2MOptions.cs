using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for M2M many-to-many splatting frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// M2M (Hu et al., CVPR 2022) uses many-to-many splatting for efficient interpolation:
/// - Many-to-many splatting: instead of the standard one-to-one backward warping (each target
///   pixel samples from one source pixel), M2M allows multiple source pixels to contribute to
///   multiple target pixels simultaneously using forward splatting with learned weights
/// - Multiple bidirectional flows: estimates K flow field pairs (forward and backward) at each
///   pyramid level, capturing multiple motion hypotheses for occluded regions and motion
///   boundaries where a single flow is ambiguous
/// - Splatting confidence: each splatted pixel carries a learned confidence weight, and the
///   final pixel value is a confidence-weighted sum of all contributions, naturally handling
///   occlusions and disocclusions
/// - Multi-scale pipeline: coarse-to-fine architecture where splatting is performed at each
///   scale, and residual corrections are added at each level
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard interpolation "pulls" each pixel from one location in the
/// source frame. M2M instead "pushes" pixels from the source to potentially multiple locations
/// in the target, which better handles cases where objects overlap or appear/disappear between
/// frames.
/// </para>
/// </remarks>
public class M2MOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public M2MOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public M2MOptions(M2MOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumFlowHypotheses = other.NumFlowHypotheses;
        NumPyramidLevels = other.NumPyramidLevels;
        NumRefineBlocks = other.NumRefineBlocks;
        SplattingRadius = other.SplattingRadius;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of flow hypotheses per pixel (K).</summary>
    /// <remarks>Multiple flows per pixel handle occlusion boundaries and ambiguous motion.</remarks>
    public int NumFlowHypotheses { get; set; } = 4;

    /// <summary>Gets or sets the number of pyramid levels for multi-scale splatting.</summary>
    public int NumPyramidLevels { get; set; } = 4;

    /// <summary>Gets or sets the number of refinement blocks at each scale level.</summary>
    public int NumRefineBlocks { get; set; } = 2;

    /// <summary>Gets or sets the splatting radius in pixels.</summary>
    /// <remarks>Controls how far a source pixel can contribute in the target frame.</remarks>
    public int SplattingRadius { get; set; } = 2;

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
