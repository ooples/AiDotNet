using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for RealBasicVSR-Sharp, the adversarial training stage of RealBasicVSR.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Chan et al., CVPR 2022, and the authors' release (open-mmlab/mmagic,
/// configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py): the RealBasicVSRNet of the first stage
/// (64 channels, 20 propagation and 20 cleaning blocks), a spectrally normalized U-Net discriminator with 64 channels,
/// L1 output and cleaning losses, a VGG19 perceptual loss and a vanilla GAN loss weighted 0.05, the output and perceptual
/// losses against the unsharp-masked target, the generator trained with Adam (0.9, 0.99) at 5e-5 and the discriminator
/// at 1e-4, SPyNet frozen, and a 0.999 moving average of the generator for inference.
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

        Seed = other.Seed;
        MidChannels = other.MidChannels;
        NumPropagationBlocks = other.NumPropagationBlocks;
        NumCleaningBlocks = other.NumCleaningBlocks;
        DiscriminatorChannels = other.DiscriminatorChannels;
        NumFrames = other.NumFrames;
        InferenceCleaningThreshold = other.InferenceCleaningThreshold;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DiscriminatorLearningRate = other.DiscriminatorLearningRate;
        PixelLossWeight = other.PixelLossWeight;
        CleaningLossWeight = other.CleaningLossWeight;
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
    public int MidChannels { get; set; } = 64;

    /// <summary>Gets or sets the residual blocks per propagation branch (20).</summary>
    public int NumPropagationBlocks { get; set; } = 20;

    /// <summary>Gets or sets the residual blocks of the cleaning module (20).</summary>
    public int NumCleaningBlocks { get; set; } = 20;

    /// <summary>Gets or sets the U-Net discriminator's base channels (64).</summary>
    public int DiscriminatorChannels { get; set; } = 64;

    /// <summary>Gets or sets the frames of a training clip (15).</summary>
    public int NumFrames { get; set; } = 15;

    /// <summary>
    /// Gets or sets the mean absolute cleaning residue, in pixel levels 0-255, below which inference stops cleaning
    /// (5, the release's test setting for this stage; training uses 255, i.e. one pass).
    /// </summary>
    public double InferenceCleaningThreshold { get; set; } = 5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the generator's Adam learning rate (5e-5).</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the discriminator's Adam learning rate (1e-4).</summary>
    public double DiscriminatorLearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight of the L1 loss on the output (1.0).</summary>
    public double PixelLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the L1 loss of the cleaned frames against the 4x area-downsampled target (1.0).</summary>
    public double CleaningLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the VGG19 perceptual loss (1.0); zero omits the VGG19 from the model.</summary>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the generator's adversarial loss (0.05).</summary>
    public double GANWeight { get; set; } = 5e-2;

    /// <summary>
    /// Gets or sets whether the output and perceptual losses score the unsharp-masked target (true); the
    /// discriminator always sees the unsharpened target.
    /// </summary>
    public bool SharpenTarget { get; set; } = true;

    /// <summary>Gets or sets the unsharp-masking weight (0.5).</summary>
    public double UnsharpWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the unsharp-masking Gaussian size (51).</summary>
    public int UnsharpKernelSize { get; set; } = 51;

    /// <summary>Gets or sets the unsharp-masking residue threshold in pixel levels (10).</summary>
    public double UnsharpThreshold { get; set; } = 10;

    /// <summary>
    /// Gets or sets the momentum of the generator's moving average used for inference (0.001); null disables it and
    /// inference uses the trained weights.
    /// </summary>
    public double? EmaMomentum { get; set; } = 0.001;

    #endregion
}
