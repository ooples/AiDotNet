using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the MGLD-VSR motion-guided latent diffusion video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// MGLD-VSR (Yang et al., 2024) integrates explicit motion guidance into latent diffusion:
/// - Motion-guided denoising: optical flow maps are injected as additional conditioning
///   into each denoising step, ensuring temporal consistency
/// - Latent diffusion: operates in a compressed latent space (VAE encoder/decoder) for
///   efficiency, with the U-Net denoiser conditioned on both the LR frames and flow
/// - Motion-aware loss: penalizes temporal inconsistency in addition to pixel accuracy
/// </para>
/// <para>
/// <b>For Beginners:</b> While most diffusion-based upscalers treat each frame independently
/// (leading to flickering), MGLD-VSR explicitly tells the AI "here is how objects moved"
/// using optical flow, so it can maintain smooth, consistent motion across frames.
/// </para>
/// </remarks>
public class MGLDVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public MGLDVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MGLDVSROptions(MGLDVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDenoisingSteps = other.NumDenoisingSteps;
        NumResBlocks = other.NumResBlocks;
        ScaleFactor = other.ScaleFactor;
        LatentDim = other.LatentDim;
        MotionGuidanceWeight = other.MotionGuidanceWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of UNet feature channels.</summary>
    public int NumFeatures { get; set; } = 128;

    /// <summary>Gets or sets the number of denoising steps.</summary>
    public int NumDenoisingSteps { get; set; } = 20;

    /// <summary>Gets or sets the number of residual blocks in the U-Net.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the latent space dimension.</summary>
    public int LatentDim { get; set; } = 4;

    /// <summary>Gets or sets the motion guidance weight.</summary>
    /// <remarks>Controls how strongly optical flow influences the denoising process.</remarks>
    public double MotionGuidanceWeight { get; set; } = 1.0;

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

    /// <summary>Gets or sets the warmup steps.</summary>
    public int WarmupSteps { get; set; } = 5000;

    #endregion
}
