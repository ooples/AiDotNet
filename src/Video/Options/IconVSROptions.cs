using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the IconVSR information-aggregation video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// IconVSR (Chan et al., CVPR 2021) extends BasicVSR with two key modules:
/// - Information-Aggregation Module: extracts features from sparsely-selected keyframes
///   and uses them to refine propagation features via a cross-attention mechanism
/// - Coupled Propagation: forward and backward branches exchange information to reduce
///   error accumulation in long sequences
/// </para>
/// <para>
/// <b>For Beginners:</b> While BasicVSR processes frames one by one, IconVSR picks
/// important "keyframes" and uses them as extra reference points. This helps
/// especially for long video sequences where errors can build up over time.
/// </para>
/// </remarks>
public class IconVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public IconVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public IconVSROptions(IconVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        KeyframeStride = other.KeyframeStride;
        NumEdemaBlocks = other.NumEdemaBlocks;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual blocks per propagation direction.</summary>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of input frames.</summary>
    public int NumFrames { get; set; } = 15;

    /// <summary>Gets or sets the stride for selecting keyframes from the sequence.</summary>
    /// <remarks>Every N-th frame is selected as a keyframe for the information-aggregation module.</remarks>
    public int KeyframeStride { get; set; } = 5;

    /// <summary>Gets or sets the number of EDVR-style deformable alignment blocks in the information-aggregation module.</summary>
    /// <remarks>These are based on the EDVR (Enhanced Deformable Video Restoration) alignment module from Chan et al.</remarks>
    public int NumEdemaBlocks { get; set; } = 5;

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

    /// <summary>Gets or sets the warmup steps for the learning rate schedule.</summary>
    public int WarmupSteps { get; set; } = 5000;

    #endregion
}
