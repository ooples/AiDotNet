using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the IART implicit resampling-based alignment transformer.
/// </summary>
/// <remarks>
/// <para>
/// IART (Kai et al., CVPR 2024 Highlight) uses implicit neural representations for alignment:
/// - Implicit resampling: instead of warping features to discrete grid positions (which
///   causes interpolation artifacts), IART uses a continuous implicit function to resample
///   features at arbitrary sub-pixel positions with learned kernels
/// - Alignment transformer: cross-attention between the reference frame and supporting
///   frames, where the attention sampling positions are offset by flow-guided implicit
///   coordinates rather than fixed grid positions
/// - Multi-scale implicit alignment: alignment at multiple feature resolutions, from
///   coarse structural alignment to fine texture-level resampling
/// - Preserves high-frequency details that grid-based warping typically blurs
/// </para>
/// <para>
/// <b>For Beginners:</b> When aligning video frames, most models "warp" one frame to match
/// another using a grid. This can blur fine details because pixel positions don't perfectly
/// line up. IART solves this by using a continuous function that can sample features at
/// any position (not just grid points), preserving sharp edges and textures that other
/// methods would smooth out.
/// </para>
/// </remarks>
public class IARTOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of alignment transformer blocks.</summary>
    public int NumTransformerBlocks { get; set; } = 6;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the number of implicit resampling scales.</summary>
    /// <remarks>Multi-scale alignment from coarse to fine.</remarks>
    public int NumScales { get; set; } = 3;

    /// <summary>Gets or sets the dimension of the implicit coordinate embedding.</summary>
    /// <remarks>Higher dimensions allow finer sub-pixel positioning.</remarks>
    public int ImplicitDim { get; set; } = 32;

    /// <summary>Gets or sets the number of residual blocks in reconstruction.</summary>
    public int NumResBlocks { get; set; } = 20;

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
