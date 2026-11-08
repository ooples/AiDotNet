using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for DualX-VSR, the dual axial spatial x temporal transformer for real-world video
/// super-resolution without motion compensation.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Cao et al. 2025 (arXiv 2506.04830), Tables 6-8 and Sec. 5.1: a 64-channel 3x3 shallow convolution,
/// 2x2 patches embedded to 1,280 channels, six vertical-temporal then six horizontal-temporal blocks, 12 heads, MLP ratio
/// 2, RoPE with base 10,000, temporal transformers of depth 2 in the embedding and the reconstruction, x4 upsampling and
/// 16-frame clips. Training is AdamW at 5e-5 for 500,000 bicubic pretraining and 150,000 real-degradation fine-tuning
/// iterations on Charbonnier loss, then 25,000 iterations at 1e-5 for the generator and discriminator on Charbonnier,
/// perceptual and GAN losses weighted 0.01, 1 and 0.005. The paper gives no code; <see cref="HeadDim"/> (64, vit-pytorch's
/// ViViT default) is the head width that reproduces its 127.95M parameters, and AdamW's unstated decay is PyTorch's 0.01.
/// </para>
/// <para>
/// <b>For Beginners:</b> DualX-VSR upscales video four times without estimating motion. These settings control how wide
/// and deep its attention is and how it is trained: first to match clean frames, then also to look realistic.
/// </para>
/// </remarks>
public class DualXVSROptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public DualXVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DualXVSROptions(DualXVSROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        NumFeatures = other.NumFeatures;
        PatchSize = other.PatchSize;
        EmbedDim = other.EmbedDim;
        NumHeads = other.NumHeads;
        HeadDim = other.HeadDim;
        MlpRatio = other.MlpRatio;
        EmbeddingTemporalDepth = other.EmbeddingTemporalDepth;
        VerticalDepth = other.VerticalDepth;
        HorizontalDepth = other.HorizontalDepth;
        ReconstructionTemporalDepth = other.ReconstructionTemporalDepth;
        RopeBase = other.RopeBase;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        DiscriminatorChannels = other.DiscriminatorChannels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        AdversarialLearningRate = other.AdversarialLearningRate;
        DiscriminatorLearningRate = other.DiscriminatorLearningRate;
        WeightDecay = other.WeightDecay;
        ReconstructionIterations = other.ReconstructionIterations;
        AdversarialPixelWeight = other.AdversarialPixelWeight;
        PerceptualWeight = other.PerceptualWeight;
        GANWeight = other.GANWeight;
        CharbonnierEpsilon = other.CharbonnierEpsilon;
    }

    #region Architecture

    /// <summary>Gets or sets the shallow convolution's channels and the per-pixel temporal transformers' width (64).</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the side of the square image patch a token covers (2).</summary>
    public int PatchSize { get; set; } = 2;

    /// <summary>Gets or sets the dual axial transformer's token width (1,280).</summary>
    public int EmbedDim { get; set; } = 1280;

    /// <summary>Gets or sets the attention heads of every transformer (12).</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the channels of each attention head (64).</summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>Gets or sets the MLP's hidden width as a multiple of the token width (2).</summary>
    public double MlpRatio { get; set; } = 2.0;

    /// <summary>Gets or sets the blocks of the input embedding's temporal transformer (2).</summary>
    public int EmbeddingTemporalDepth { get; set; } = 2;

    /// <summary>Gets or sets the vertical-temporal attention blocks (6).</summary>
    public int VerticalDepth { get; set; } = 6;

    /// <summary>Gets or sets the horizontal-temporal attention blocks, which follow the vertical ones (6).</summary>
    public int HorizontalDepth { get; set; } = 6;

    /// <summary>Gets or sets the blocks of the reconstruction's temporal transformer (2).</summary>
    public int ReconstructionTemporalDepth { get; set; } = 2;

    /// <summary>Gets or sets the rotary position embedding's base b (10,000).</summary>
    public double RopeBase { get; set; } = 10000.0;

    /// <summary>Gets or sets the spatial upscaling factor, a power of two (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the frames of a clip (16).</summary>
    public int NumFrames { get; set; } = 16;

    /// <summary>Gets or sets the discriminator's base width (64, RealBasicVSR's U-Net discriminator).</summary>
    public int DiscriminatorChannels { get; set; } = 64;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the AdamW rate of the pretraining and fine-tuning stages (5e-5).</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the generator's AdamW rate in the GAN stage (1e-5).</summary>
    public double AdversarialLearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the discriminator's AdamW rate (1e-5).</summary>
    public double DiscriminatorLearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets AdamW's decoupled weight decay (0.01, PyTorch's default; the paper states none).</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the Charbonnier-only iterations before the GAN stage (650,000: 500,000 pretraining plus 150,000
    /// fine-tuning, which differ only in their data).
    /// </summary>
    public int ReconstructionIterations { get; set; } = 650000;

    /// <summary>Gets or sets the Charbonnier weight in the GAN stage (0.01).</summary>
    public double AdversarialPixelWeight { get; set; } = 0.01;

    /// <summary>Gets or sets the perceptual loss weight in the GAN stage (1; 0 leaves the VGG19 out).</summary>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the GAN loss weight in the GAN stage (0.005).</summary>
    public double GANWeight { get; set; } = 0.005;

    /// <summary>Gets or sets the Charbonnier ε (1e-6, i.e. sqrt(d² + 1e-12)).</summary>
    public double CharbonnierEpsilon { get; set; } = 1e-6;

    #endregion
}
