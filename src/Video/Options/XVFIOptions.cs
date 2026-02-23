using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for XVFI extreme video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// XVFI (Sim et al., ICCV 2021) handles extreme motion for high-FPS video:
/// - Extreme motion handling: designed for 4K/8K video with very large frame-to-frame
///   displacements (100+ pixels), far beyond what standard flow networks can handle
/// - Complementary flow: estimates both global (affine) and local (dense) optical flow fields,
///   combining them with learned blending weights so global flow handles camera motion and
///   local flow handles object motion
/// - Multi-scale architecture: a 7-level feature pyramid with flow estimation at each scale,
///   starting from 1/64 resolution for very large motions and refining up to full resolution
/// - Bilinear flow upsampling: uses learned bilinear upsampling kernels (not fixed bilinear
///   interpolation) to upsample flow fields between pyramid levels, preserving sharp motion
///   boundaries during upsampling
/// </para>
/// <para>
/// <b>For Beginners:</b> XVFI is designed for extreme cases: very high resolution video (4K/8K)
/// where objects move very far between frames. It uses a multi-level approach that first captures
/// big movements, then progressively adds fine detail, enabling frame interpolation even when
/// objects move hundreds of pixels between frames.
/// </para>
/// </remarks>
public class XVFIOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public XVFIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public XVFIOptions(XVFIOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumPyramidLevels = other.NumPyramidLevels;
        NumResBlocks = other.NumResBlocks;
        NumAffineParams = other.NumAffineParams;
        UseComplementaryFlow = other.UseComplementaryFlow;
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

    /// <summary>Gets or sets the number of pyramid levels.</summary>
    public int NumPyramidLevels { get; set; } = 7;

    /// <summary>Gets or sets the number of residual blocks per level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the number of global flow affine parameters.</summary>
    public int NumAffineParams { get; set; } = 6;

    /// <summary>Gets or sets whether to use complementary flow (global + local).</summary>
    public bool UseComplementaryFlow { get; set; } = true;

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
