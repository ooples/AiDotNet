using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the BasicVSR bidirectional recurrent video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// BasicVSR (Chan et al., CVPR 2021) establishes the essential components for video SR:
/// - Bidirectional recurrent propagation: forward and backward passes across frames
/// - Optical flow-based alignment: SpyNet estimates motion between adjacent frames
/// - Residual feature refinement: 30 residual blocks per propagation direction
/// </para>
/// <para>
/// <b>For Beginners:</b> BasicVSR treats video frames like a sequence. It processes frames
/// both forward in time and backward, so each frame benefits from information in both
/// directions. Optical flow tells the model how pixels moved between frames, so it can
/// align them before combining their features for higher resolution output.
/// </para>
/// </remarks>
public class BasicVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public BasicVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BasicVSROptions(BasicVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        MidChannels = other.MidChannels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels in the propagation branches.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual blocks per propagation direction.</summary>
    /// <remarks>The paper uses 30 blocks for the base model.</remarks>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of input frames for bidirectional propagation.</summary>
    public int NumFrames { get; set; } = 15;

    /// <summary>Gets or sets the mid-channel dimension in residual blocks.</summary>
    public int MidChannels { get; set; } = 64;

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

    /// <summary>Gets or sets the number of warmup iterations.</summary>
    public int WarmupSteps { get; set; } = 5000;

    #endregion
}
