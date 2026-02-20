using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for MoMo momentum diffusion model for bi-directional flow.
/// </summary>
/// <remarks>
/// <para>
/// MoMo (2024) is the first diffusion model for bi-directional optical flow in VFI:
/// - Flow diffusion: instead of directly regressing optical flow from a CNN (which produces
///   over-smoothed flow at boundaries), MoMo uses a denoising diffusion model to generate
///   bi-directional flow fields, capturing sharper motion boundaries and multi-modal flow
/// - Momentum-based flow modeling: incorporates a momentum prior that biases flow generation
///   toward physically plausible motions, reducing artifacts from unrealistic flow predictions
/// - Joint bi-directional generation: generates forward (t0 to t) and backward (t1 to t)
///   flows simultaneously in a single diffusion process, ensuring temporal consistency between
///   the two flow fields
/// - Flow-to-frame synthesis: the generated flows are used for backward warping with learned
///   occlusion masks and residual refinement to produce the final interpolated frame
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods estimate motion (optical flow) using
/// direct prediction, which can be blurry at object edges. MoMo instead uses a generative AI
/// model to create sharper, more accurate motion fields, then uses those flows to produce
/// the intermediate frame. Think of it as using AI to "draw" better motion maps.
/// </para>
/// </remarks>
public class MoMoOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of diffusion denoising steps.</summary>
    public int NumDiffusionSteps { get; set; } = 25;

    /// <summary>Gets or sets the number of U-Net residual blocks per level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the number of attention heads in the denoising network.</summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>Gets or sets the momentum coefficient for the flow prior.</summary>
    /// <remarks>Controls how strongly the model biases toward physically plausible flow.</remarks>
    public double MomentumCoefficient { get; set; } = 0.9;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
