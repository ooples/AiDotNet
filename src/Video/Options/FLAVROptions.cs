using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FLAVR flow-agnostic video representations model.
/// </summary>
/// <remarks>
/// <para>
/// FLAVR (Kalluri et al., CVPR 2023) uses 3D convolutions for flow-free interpolation:
/// - 3D spatio-temporal convolutions: processes multiple input frames simultaneously using
///   3D (space + time) convolutions that capture temporal relationships without explicit
///   optical flow estimation
/// - 3D encoder-decoder: a U-Net style architecture where the encoder uses strided 3D
///   convolutions to downsample in both space and time, and the decoder uses transposed
///   3D convolutions to upsample back to full resolution
/// - Multi-frame input: takes 4 input frames (2 before and 2 after the target) for richer
///   temporal context, unlike 2-frame methods
/// - Direct synthesis: directly outputs the target frame pixels without intermediate flow
///   or warping operations, avoiding flow estimation errors entirely
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods first figure out how objects move
/// (optical flow), then use that to warp frames. FLAVR skips the flow step entirely by using
/// 3D convolutions that "see" multiple frames at once and directly paint the intermediate
/// frame. This makes it faster and avoids the ghosting artifacts that come from bad flow
/// estimates, especially in scenes with transparent objects or repetitive patterns.
/// </para>
/// </remarks>
public class FLAVROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public FLAVROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FLAVROptions(FLAVROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        NumLevels = other.NumLevels;
        NumInputFrames = other.NumInputFrames;
        TemporalKernelSize = other.TemporalKernelSize;
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

    /// <summary>Gets or sets the number of 3D residual blocks per encoder level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the number of encoder/decoder levels.</summary>
    /// <remarks>Each level halves/doubles spatial resolution. Paper uses 4 levels.</remarks>
    public int NumLevels { get; set; } = 4;

    /// <summary>Gets or sets the number of input frames (temporal context window).</summary>
    /// <remarks>FLAVR uses 4 frames: 2 past + 2 future for maximum temporal context.</remarks>
    public int NumInputFrames { get; set; } = 4;

    /// <summary>Gets or sets the temporal kernel size for 3D convolutions.</summary>
    public int TemporalKernelSize { get; set; } = 3;

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
