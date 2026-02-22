using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the DynamiCrafter video diffusion interpolation model.
/// </summary>
/// <remarks>
/// <para>
/// DynamiCrafter (2024) uses video diffusion priors for frame interpolation:
/// - Video diffusion backbone: adapts a pre-trained text-to-video diffusion model (e.g.,
///   Stable Video Diffusion) for the interpolation task, leveraging its learned motion priors
/// - First/last frame conditioning: the diffusion process is conditioned on both the first
///   and last frames using CLIP image embeddings injected via cross-attention, ensuring the
///   generated intermediate frames are temporally consistent with both endpoints
/// - Noise schedule adaptation: modified diffusion noise schedule that biases early denoising
///   steps toward global motion consistency and later steps toward local detail refinement
/// - Temporal attention: 3D self-attention across generated frames ensures smooth motion
///   transitions without flickering or temporal discontinuities
/// </para>
/// <para>
/// <b>For Beginners:</b> DynamiCrafter uses an AI image/video generator (diffusion model)
/// that already knows how things move in the real world. Given a start frame and end frame,
/// it gradually "imagines" what happens in between, producing natural-looking intermediate
/// frames with realistic motion, lighting changes, and object interactions.
/// </para>
/// </remarks>
public class DynamiCrafterOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public DynamiCrafterOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DynamiCrafterOptions(DynamiCrafterOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDiffusionSteps = other.NumDiffusionSteps;
        NumResBlocks = other.NumResBlocks;
        NumHeads = other.NumHeads;
        GuidanceScale = other.GuidanceScale;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels in the UNet.</summary>
    public int NumFeatures { get; set; } = 128;

    /// <summary>Gets or sets the number of diffusion timesteps during inference.</summary>
    /// <remarks>More steps = higher quality but slower. Paper uses 50 DDIM steps.</remarks>
    public int NumDiffusionSteps { get; set; } = 50;

    /// <summary>Gets or sets the number of UNet residual blocks per level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the number of attention heads in temporal attention.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the classifier-free guidance scale.</summary>
    /// <remarks>Higher values make output more faithful to conditioning but less diverse.</remarks>
    public double GuidanceScale { get; set; } = 7.5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
