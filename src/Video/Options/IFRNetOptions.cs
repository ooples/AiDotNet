using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the IFRNet intermediate feature refine network.
/// </summary>
/// <remarks>
/// <para>
/// IFRNet (Kong et al., CVPR 2022) uses coarse-to-fine intermediate feature refinement:
/// - Encoder-decoder with skip connections: shared encoder extracts multi-scale features
///   from both input frames, decoder progressively refines the interpolation result from
///   coarsest to finest scale
/// - Intermediate feature refinement (IFR): at each decoder level, instead of refining the
///   optical flow, IFRNet directly refines the intermediate features of the target frame,
///   avoiding error accumulation from flow estimation
/// - Coarse-to-fine architecture: 3-level pyramid where each level operates at half the
///   resolution of the next, with learned upsampling between levels
/// - Task-oriented flow: optical flow is used only as an initial guide for feature warping,
///   then discarded in favor of direct feature refinement
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods first estimate motion (optical flow),
/// then use it to warp frames. If the flow is wrong, the result is wrong. IFRNet takes a
/// different approach: it starts with a rough flow estimate, uses it to get initial features,
/// then directly refines those features to produce the final frame. This is more forgiving
/// of flow errors and produces sharper results.
/// </para>
/// </remarks>
public class IFRNetOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels at the finest level.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of refinement blocks per decoder level.</summary>
    /// <remarks>More blocks allow finer detail recovery. Paper uses 4 per level.</remarks>
    public int NumRefineBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of pyramid levels.</summary>
    /// <remarks>3 levels = 1x, 1/2x, 1/4x resolution. Higher is better for large motion.</remarks>
    public int NumPyramidLevels { get; set; } = 3;

    /// <summary>Gets or sets whether to use task-oriented flow initialization.</summary>
    public bool UseTaskOrientedFlow { get; set; } = true;

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
