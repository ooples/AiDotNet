using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the TTVSR trajectory-aware transformer for video SR.
/// </summary>
/// <remarks>
/// <para>
/// TTVSR (Liu et al., ECCV 2022) tracks feature trajectories across frames:
/// - Trajectory-aware attention: instead of attending to fixed spatial locations,
///   attention follows estimated motion trajectories so features are gathered along
///   the path an object actually moved
/// - Cross-scale feature tokenization: visual tokens are extracted at multiple scales
///   and fused, capturing both fine texture and coarse structure
/// - Location map: a learned spatial map that helps the transformer locate the
///   trajectory routing positions across frames
/// - Long-range temporal modeling: trajectories span the full video length, not just
///   adjacent frames, enabling information flow from distant frames
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine following a ball as it moves across frames. Instead
/// of looking at the same spot in every frame, TTVSR tracks where objects actually
/// go and gathers information along their path. This is much more effective than
/// fixed-position alignment because real video has complex motion.
/// </para>
/// </remarks>
public class TTVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TTVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TTVSROptions(TTVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumTransformerBlocks = other.NumTransformerBlocks;
        ScaleFactor = other.ScaleFactor;
        TrajectoryLength = other.TrajectoryLength;
        NumHeads = other.NumHeads;
        NumScales = other.NumScales;
        NumResBlocks = other.NumResBlocks;
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

    /// <summary>Gets or sets the number of trajectory-aware transformer blocks.</summary>
    public int NumTransformerBlocks { get; set; } = 6;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the trajectory length (number of frames tracked per trajectory).</summary>
    /// <remarks>Longer trajectories capture more temporal context but use more memory.</remarks>
    public int TrajectoryLength { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>Gets or sets the number of cross-scale feature tokenization levels.</summary>
    /// <remarks>Features are extracted at this many spatial scales and fused.</remarks>
    public int NumScales { get; set; } = 3;

    /// <summary>Gets or sets the number of residual blocks in the reconstruction module.</summary>
    public int NumResBlocks { get; set; } = 30;

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
