using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for Real-ESRGAN applied to video, frame by frame.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Wang et al. 2021 and the authors' release (xinntao/Real-ESRGAN, RealESRGAN_x4plus,
/// options/train_realesrgan_x4plus.yml, and inference_realesrgan_video.py, which upscales each frame independently):
/// an RRDBNet generator with 64 features, 23 residual-in-residual dense blocks and growth 32 at x4, a spectrally
/// normalized U-Net discriminator with 64 channels, an L1 loss, a VGG19 perceptual loss and a vanilla GAN loss weighted
/// 0.1, the L1 and perceptual losses against the unsharp-masked target, both networks trained with Adam (0.9, 0.99) at
/// 1e-4, and a 0.999 moving average of the generator for inference.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real-ESRGAN is one of the most popular practical upscaling tools. It is trained on
/// images degraded the way real photos and videos are (blur, noise, resizing, compression), so it cleans those
/// artifacts while it upscales. For video it simply upscales every frame.
/// </para>
/// </remarks>
public class RealESRGANVideoOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RealESRGANVideoOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RealESRGANVideoOptions(RealESRGANVideoOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        NumFeatures = other.NumFeatures;
        NumRRDBBlocks = other.NumRRDBBlocks;
        GrowthChannels = other.GrowthChannels;
        DiscriminatorChannels = other.DiscriminatorChannels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DiscriminatorLearningRate = other.DiscriminatorLearningRate;
        PixelLossWeight = other.PixelLossWeight;
        PerceptualWeight = other.PerceptualWeight;
        GANWeight = other.GANWeight;
        SharpenTarget = other.SharpenTarget;
        UnsharpWeight = other.UnsharpWeight;
        UnsharpKernelSize = other.UnsharpKernelSize;
        UnsharpThreshold = other.UnsharpThreshold;
        EmaMomentum = other.EmaMomentum;
    }

    #region Architecture

    /// <summary>Gets or sets the generator's feature channels (64).</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the residual-in-residual dense blocks (23).</summary>
    public int NumRRDBBlocks { get; set; } = 23;

    /// <summary>Gets or sets the dense blocks' growth channels (32).</summary>
    public int GrowthChannels { get; set; } = 32;

    /// <summary>Gets or sets the U-Net discriminator's base channels (64).</summary>
    public int DiscriminatorChannels { get; set; } = 64;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the generator's Adam learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the discriminator's Adam learning rate (1e-4).</summary>
    public double DiscriminatorLearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight of the L1 loss (1.0).</summary>
    public double PixelLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the VGG19 perceptual loss (1.0); zero omits the VGG19 from the model.</summary>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the generator's adversarial loss (0.1).</summary>
    public double GANWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether the L1 and perceptual losses score the unsharp-masked target (true, l1_gt_usm and
    /// percep_gt_usm); the discriminator always sees the unsharpened target.
    /// </summary>
    public bool SharpenTarget { get; set; } = true;

    /// <summary>Gets or sets the unsharp-masking weight (0.5).</summary>
    public double UnsharpWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the unsharp-masking Gaussian size (51).</summary>
    public int UnsharpKernelSize { get; set; } = 51;

    /// <summary>Gets or sets the unsharp-masking residue threshold in pixel levels (10).</summary>
    public double UnsharpThreshold { get; set; } = 10;

    /// <summary>
    /// Gets or sets the momentum of the generator's moving average used for inference (0.001, ema_decay 0.999); null
    /// disables it and inference uses the trained weights.
    /// </summary>
    public double? EmaMomentum { get; set; } = 0.001;

    #endregion
}
