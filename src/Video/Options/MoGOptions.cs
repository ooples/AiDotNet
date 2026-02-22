using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for MoG motion-aware generative frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// MoG (2025) combines flow estimation with diffusion-based generation:
/// - Motion-aware conditioning: first estimates bidirectional optical flow using an EMA-VFI-style
///   flow network, then uses the estimated flows as spatial conditioning for a diffusion model
///   rather than directly warping frames
/// - Flow-conditioned diffusion: the denoising U-Net receives concatenated flow maps as
///   additional input channels, guiding the diffusion process to generate motion-consistent
///   intermediate frames with fine texture details
/// - Generative refinement: instead of blending warped frames (which can produce ghosting),
///   the diffusion model generates the intermediate frame from scratch, conditioned on the
///   input frames and estimated motion, producing sharp results even in occluded regions
/// - Progressive denoising: multi-step denoising with motion-aware noise scheduling that
///   preserves motion coherence in early steps and refines textures in later steps
/// </para>
/// <para>
/// <b>For Beginners:</b> MoG combines two approaches: first it figures out how things move
/// (optical flow), then uses a generative AI model (diffusion) to "paint" the intermediate
/// frame guided by that motion information. This produces sharper results than just blending
/// warped frames, especially for complex motions.
/// </para>
/// </remarks>
public class MoGOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public MoGOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MoGOptions(MoGOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDiffusionSteps = other.NumDiffusionSteps;
        NumFlowScales = other.NumFlowScales;
        NumResBlocks = other.NumResBlocks;
        GuidanceScale = other.GuidanceScale;
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

    /// <summary>Gets or sets the number of diffusion denoising steps during inference.</summary>
    public int NumDiffusionSteps { get; set; } = 20;

    /// <summary>Gets or sets the number of flow estimation scales.</summary>
    public int NumFlowScales { get; set; } = 3;

    /// <summary>Gets or sets the number of U-Net residual blocks per level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the guidance scale for classifier-free guidance.</summary>
    /// <remarks>Higher values increase motion fidelity but may reduce diversity.</remarks>
    public double GuidanceScale { get; set; } = 3.0;

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
