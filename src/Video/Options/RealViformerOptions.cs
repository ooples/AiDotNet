using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for RealViformer, the channel-attention recurrent transformer for real-world video
/// super-resolution.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Zhang and Yao, ECCV 2024, Sec. 4.4, and the authors' release (Yuehan717/RealViformer,
/// inference_realviformer.py): 48 channels, [2, 3, 4] transformer blocks over three levels plus one refinement block,
/// [1, 2, 4] heads, GDFN expansion 2.66, a two-head CAF, attention squeezed by 4 with the channel mask, 15-frame training
/// clips, SPyNet frozen for 5,000 iterations, 300,000 iterations of Charbonnier + SSIM and then perceptual and GAN losses
/// (weights 1, 0.001, 1, 0.005) with RealBasicVSR's discriminator. The optimizer is not in the public paper, so the rates
/// are RealBasicVSR's: Adam (0.9, 0.99) at 1e-4, then 5e-5 for the generator and 1e-4 for the discriminator.
/// </para>
/// <para>
/// <b>For Beginners:</b> RealViformer upscales real, messy video four times. These settings control how wide and deep
/// the network is and how it is trained: first to match the clean frames, then to also look realistic.
/// </para>
/// </remarks>
public class RealViformerOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RealViformerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RealViformerOptions(RealViformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        NumFeatures = other.NumFeatures;
        NumBlocks = (int[])other.NumBlocks.Clone();
        Heads = (int[])other.Heads.Clone();
        FfnExpansionFactor = other.FfnExpansionFactor;
        MergeHeads = other.MergeHeads;
        SqueezeFactor = other.SqueezeFactor;
        ChannelMask = other.ChannelMask;
        DiscriminatorChannels = other.DiscriminatorChannels;
        NumFrames = other.NumFrames;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        AdversarialLearningRate = other.AdversarialLearningRate;
        DiscriminatorLearningRate = other.DiscriminatorLearningRate;
        ReconstructionIterations = other.ReconstructionIterations;
        FlowFreezeIterations = other.FlowFreezeIterations;
        CharbonnierWeight = other.CharbonnierWeight;
        SsimWeight = other.SsimWeight;
        PerceptualWeight = other.PerceptualWeight;
        GANWeight = other.GANWeight;
        CharbonnierEpsilon = other.CharbonnierEpsilon;
        EmaMomentum = other.EmaMomentum;
    }

    #region Architecture

    /// <summary>Gets or sets the first level's channels (48; the levels below have 96 and 192).</summary>
    public int NumFeatures { get; set; } = 48;

    /// <summary>Gets or sets the transformer blocks of levels 1-3 and the refinement ([2, 3, 4, 1]).</summary>
    public int[] NumBlocks { get; set; } = { 2, 3, 4, 1 };

    /// <summary>Gets or sets the attention heads of levels 1-3 ([1, 2, 4]).</summary>
    public int[] Heads { get; set; } = { 1, 2, 4 };

    /// <summary>Gets or sets the GDFN's hidden width as a multiple of the channels (2.66).</summary>
    public double FfnExpansionFactor { get; set; } = 2.66;

    /// <summary>Gets or sets the heads of the CAF that merges the hidden state (2).</summary>
    public int MergeHeads { get; set; } = 2;

    /// <summary>Gets or sets the factor the attention squeezes channels by (4; 1 disables the squeeze).</summary>
    public int SqueezeFactor { get; set; } = 4;

    /// <summary>Gets or sets whether the attention re-weights its channels with the learned mask (true).</summary>
    public bool ChannelMask { get; set; } = true;

    /// <summary>Gets or sets the discriminator's base width (64, RealBasicVSR's U-Net discriminator).</summary>
    public int DiscriminatorChannels { get; set; } = 64;

    /// <summary>Gets or sets the frames of a training clip (15).</summary>
    public int NumFrames { get; set; } = 15;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the first stage's Adam rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the second stage's generator Adam rate (5e-5).</summary>
    public double AdversarialLearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the discriminator's Adam rate (1e-4).</summary>
    public double DiscriminatorLearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the iterations of the Charbonnier + SSIM stage before perceptual and GAN losses join (300,000).</summary>
    public int ReconstructionIterations { get; set; } = 300000;

    /// <summary>Gets or sets the iterations SPyNet stays fixed at the start of training (5,000).</summary>
    public int FlowFreezeIterations { get; set; } = 5000;

    /// <summary>Gets or sets the Charbonnier loss weight (1).</summary>
    public double CharbonnierWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the SSIM loss weight (0.001).</summary>
    public double SsimWeight { get; set; } = 0.001;

    /// <summary>Gets or sets the perceptual loss weight in the second stage (1; 0 leaves the VGG19 out).</summary>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the GAN loss weight in the second stage (0.005).</summary>
    public double GANWeight { get; set; } = 0.005;

    /// <summary>Gets or sets the Charbonnier ε (1e-6, i.e. sqrt(d² + 1e-12)).</summary>
    public double CharbonnierEpsilon { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the momentum of a moving average of the weights used for inference; null (the paper reports none)
    /// uses the trained weights.
    /// </summary>
    public double? EmaMomentum { get; set; }

    #endregion
}
