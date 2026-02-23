using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for SoftSplat softmax splatting for video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// SoftSplat (Niklaus &amp; Liu, CVPR 2020) uses softmax splatting for forward warping:
/// - Forward warping with softmax: instead of backward warping (which creates holes at
///   disocclusions), SoftSplat uses forward warping where source pixels are "splatted" to
///   target positions, with conflicts resolved via softmax weighting
/// - Importance metric Z: each source pixel carries a learned importance metric Z that controls
///   its softmax weight during splatting, allowing the model to automatically learn that
///   foreground (closer) objects should occlude background (farther) objects
/// - Feature-space splatting: splatting is performed not on raw pixels but on deep feature maps,
///   allowing the synthesis network to work with rich feature representations and produce
///   higher-quality output
/// - Synthesis network: a GridNet-style synthesis network takes the splatted feature maps and
///   produces the final interpolated frame, handling residual refinement and artifact removal
/// </para>
/// <para>
/// <b>For Beginners:</b> When you push pixels from a source frame to a new position (forward
/// warping), multiple pixels might land in the same spot. SoftSplat uses a smart voting system
/// (softmax) where each pixel gets a learned "importance score" to decide which pixel wins
/// when there's a conflict, naturally handling which objects appear in front of others.
/// </para>
/// </remarks>
public class SoftSplatOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public SoftSplatOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SoftSplatOptions(SoftSplatOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumGridNetLevels = other.NumGridNetLevels;
        NumResBlocksPerRow = other.NumResBlocksPerRow;
        NumFeatureBlocks = other.NumFeatureBlocks;
        UseImportanceMetric = other.UseImportanceMetric;
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

    /// <summary>Gets or sets the number of GridNet synthesis levels.</summary>
    public int NumGridNetLevels { get; set; } = 3;

    /// <summary>Gets or sets the number of residual blocks per GridNet row.</summary>
    public int NumResBlocksPerRow { get; set; } = 3;

    /// <summary>Gets or sets the feature extraction depth (number of VGG-like blocks).</summary>
    public int NumFeatureBlocks { get; set; } = 5;

    /// <summary>Gets or sets whether to use the learned importance metric Z.</summary>
    /// <remarks>When false, falls back to average splatting (no occlusion reasoning).</remarks>
    public bool UseImportanceMetric { get; set; } = true;

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
