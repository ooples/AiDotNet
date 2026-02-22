using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RealBasicVSR-Sharp perceptually-optimized video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// RealBasicVSR-Sharp (Chan et al., CVPR 2022) is the perceptual variant of RealBasicVSR:
/// - Uses perceptual loss (VGG feature matching) instead of pixel-only L1 loss
/// - Adds GAN discriminator loss for sharper, more realistic textures
/// - Same pre-cleaning module and BasicVSR backbone as the base variant
/// - Produces visually sharper results at the cost of slightly lower PSNR
/// </para>
/// <para>
/// <b>For Beginners:</b> The "Sharp" variant trades mathematical accuracy for visual quality.
/// While the base RealBasicVSR optimizes for pixel-perfect reconstruction (high PSNR),
/// this variant uses perceptual and adversarial losses to produce results that look sharper
/// and more natural to human eyes, even if individual pixels differ slightly.
/// </para>
/// </remarks>
public class RealBasicVSRSharpOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RealBasicVSRSharpOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RealBasicVSRSharpOptions(RealBasicVSRSharpOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        CleaningModuleBlocks = other.CleaningModuleBlocks;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        PerceptualWeight = other.PerceptualWeight;
        GANWeight = other.GANWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
    }

    #region Architecture

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual blocks in the BasicVSR backbone.</summary>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the number of residual blocks in the pre-cleaning module.</summary>
    public int CleaningModuleBlocks { get; set; } = 20;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of input frames.</summary>
    public int NumFrames { get; set; } = 15;

    /// <summary>Gets or sets the perceptual loss weight.</summary>
    /// <remarks>Weight for VGG feature matching loss. Higher values produce sharper but potentially less accurate results.</remarks>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the GAN discriminator loss weight.</summary>
    public double GANWeight { get; set; } = 0.1;

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
