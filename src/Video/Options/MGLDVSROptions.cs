using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for MGLD-VSR, motion-guided latent diffusion for real-world video super-resolution.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Yang et al. 2024 (ECCV, arXiv 2312.00853) and the release (IanYeung/MGLD-VSR,
/// <c>mgldvsr_512_realbasicvsr_deg.yaml</c> and <c>video_autoencoder_kl_64x64x4_resi.yaml</c>): Stable Diffusion 2.1-512
/// as the denoiser (320 wide, multipliers 1, 2, 4, 4, two residual blocks a level, 64-wide heads, a 1,024-wide OpenCLIP
/// context of 77 tokens) with SPADE on a 256-channel time-aware structure condition (256 wide, multipliers 1, 1, 2, 2,
/// four heads) and temporal modules in the middle block over five-frame clips; the linear DDPM schedule (0.00085 to
/// 0.012 over 1,000 steps); SD's KL autoencoder (128 wide, multipliers 1, 2, 4, 4) whose decoder gains CFW modules and
/// temporal convolutions; RAFT for the motion guidance at scale −10; and the decoder's losses (L1, LPIPS 0.5, the frame
/// difference 0.5, the edge-weighted warping loss 0.5, and a PatchGAN weighted 0.025 adaptively from step 501).
/// </para>
/// <para>
/// <b>For Beginners:</b> MGLD-VSR sharpens video with an image diffusion model. While it removes noise it checks how
/// objects move between frames (optical flow) and nudges the frames to agree, so the result does not flicker.
/// </para>
/// </remarks>
public class MGLDVSROptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MGLDVSROptions()
    {
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MGLDVSROptions(MGLDVSROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        UNetChannels = other.UNetChannels;
        UNetMultipliers = (int[])other.UNetMultipliers.Clone();
        UNetResBlocks = other.UNetResBlocks;
        AttentionLevels = other.AttentionLevels;
        HeadChannels = other.HeadChannels;
        ContextDim = other.ContextDim;
        ContextLength = other.ContextLength;
        NumFrames = other.NumFrames;
        StructureChannels = other.StructureChannels;
        StructureMultipliers = (int[])other.StructureMultipliers.Clone();
        StructureResBlocks = other.StructureResBlocks;
        StructureHeads = other.StructureHeads;
        VaeChannels = other.VaeChannels;
        VaeMultipliers = (int[])other.VaeMultipliers.Clone();
        VaeResBlocks = other.VaeResBlocks;
        FuseBlocks = other.FuseBlocks;
        LatentScale = other.LatentScale;
        NumTrainTimesteps = other.NumTrainTimesteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        SamplingSteps = other.SamplingSteps;
        GuidanceScale = other.GuidanceScale;
        FlowIterations = other.FlowIterations;
        FusionWeight = other.FusionWeight;
        TrainingFusionWeight = other.TrainingFusionWeight;
        ScaleFactor = other.ScaleFactor;
        MinimumSide = other.MinimumSide;
        ColorFix = other.ColorFix;
        LearningRate = other.LearningRate;
        DecoderLearningRate = other.DecoderLearningRate;
        DiffusionIterations = other.DiffusionIterations;
        PerceptualWeight = other.PerceptualWeight;
        DifferenceWeight = other.DifferenceWeight;
        TemporalWeight = other.TemporalWeight;
        EdgeWeight = other.EdgeWeight;
        DiscriminatorWeight = other.DiscriminatorWeight;
        DiscriminatorStart = other.DiscriminatorStart;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Denoiser

    /// <summary>Gets or sets the UNet's base width (SD 2.1: 320). Every width must split into 32 groups.</summary>
    public int UNetChannels { get; set; } = 320;

    /// <summary>Gets or sets the UNet's width multipliers (1, 2, 4, 4).</summary>
    public int[] UNetMultipliers { get; set; } = { 1, 2, 4, 4 };

    /// <summary>Gets or sets the residual blocks a UNet level (2).</summary>
    public int UNetResBlocks { get; set; } = 2;

    /// <summary>Gets or sets how many leading levels carry spatial transformers (attention resolutions 4, 2, 1: three).</summary>
    public int AttentionLevels { get; set; } = 3;

    /// <summary>Gets or sets the attention head width (64).</summary>
    public int HeadChannels { get; set; } = 64;

    /// <summary>Gets or sets the text context's width (OpenCLIP ViT-H penultimate layer: 1,024).</summary>
    public int ContextDim { get; set; } = 1024;

    /// <summary>Gets or sets the text context's token count (77). The release always conditions on the empty prompt.</summary>
    public int ContextLength { get; set; } = 77;

    /// <summary>Gets or sets the frames a clip the temporal modules span (5).</summary>
    public int NumFrames { get; set; } = 5;

    /// <summary>Gets or sets the structure encoder's width and its condition's channels (256).</summary>
    public int StructureChannels { get; set; } = 256;

    /// <summary>Gets or sets the structure encoder's width multipliers (1, 1, 2, 2).</summary>
    public int[] StructureMultipliers { get; set; } = { 1, 1, 2, 2 };

    /// <summary>Gets or sets the structure encoder's residual blocks a level (2).</summary>
    public int StructureResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the structure encoder's attention heads (4).</summary>
    public int StructureHeads { get; set; } = 4;

    #endregion

    #region Autoencoder

    /// <summary>Gets or sets the KL autoencoder's base width (128).</summary>
    public int VaeChannels { get; set; } = 128;

    /// <summary>Gets or sets the autoencoder's width multipliers (1, 2, 4, 4; x8 compression).</summary>
    public int[] VaeMultipliers { get; set; } = { 1, 2, 4, 4 };

    /// <summary>Gets or sets the encoder's residual blocks a level (2; the decoder has one more).</summary>
    public int VaeResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the residual dense blocks in each CFW module (2).</summary>
    public int FuseBlocks { get; set; } = 2;

    /// <summary>Gets or sets the latent scale factor (SD's 0.18215).</summary>
    public double LatentScale { get; set; } = 0.18215;

    #endregion

    #region Diffusion and inference

    /// <summary>Gets or sets the DDPM training timesteps (1,000).</summary>
    public int NumTrainTimesteps { get; set; } = 1000;

    /// <summary>Gets or sets the first β of the linear (square-root-spaced) schedule (0.00085).</summary>
    public double BetaStart { get; set; } = 0.00085;

    /// <summary>Gets or sets the last β (0.012).</summary>
    public double BetaEnd { get; set; } = 0.012;

    /// <summary>Gets or sets the DDPM sampling steps, evenly respaced (the release's default: 1,000).</summary>
    public int SamplingSteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the motion guidance scale g (−10): after each step the latents move by −g·log σ²_t·∇L, with L the
    /// occlusion-masked warping error between neighbouring latents.
    /// </summary>
    public double GuidanceScale { get; set; } = -10.0;

    /// <summary>Gets or sets RAFT's refinement iterations (RAFT_SR's 10).</summary>
    public int FlowIterations { get; set; } = 10;

    /// <summary>Gets or sets the CFW fusion weight at inference (the release's dec_w, 0.5).</summary>
    public double FusionWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the CFW fusion weight while the decoder trains (1.0).</summary>
    public double TrainingFusionWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the upscaling factor (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>
    /// Gets or sets the shortest side the frames are upsampled to before diffusion (512, SD 2.1's resolution); the result
    /// is resized back to the scale factor. Zero upsamples by the scale factor alone.
    /// </summary>
    public int MinimumSide { get; set; } = 512;

    /// <summary>Gets or sets whether each output frame's color statistics are matched to the LQ frame (AdaIN, the paper's).</summary>
    public bool ColorFix { get; set; } = true;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the ONNX model path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the diffusion stage's AdamW learning rate (the config's base rate, 5e-5).</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the decoder stage's Adam learning rate for the decoder and the discriminator (5e-5, β 0.5 and 0.9).</summary>
    public double DecoderLearningRate { get; set; } = 5e-5;

    /// <summary>
    /// Gets or sets the diffusion stage's iterations before the decoder stage. The release trains the two stages as
    /// separate runs and states no count; the default leaves the decoder stage to a later call.
    /// </summary>
    public int DiffusionIterations { get; set; } = 100000;

    /// <summary>Gets or sets the decoder's LPIPS weight (0.5).</summary>
    public double PerceptualWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the decoder's frame-difference loss weight (0.5).</summary>
    public double DifferenceWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the decoder's warping loss weight (0.5).</summary>
    public double TemporalWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the warping loss's edge weight w in 1 + w·Sobel(ground truth) (3).</summary>
    public double EdgeWeight { get; set; } = 3.0;

    /// <summary>Gets or sets the adversarial weight on the adaptive GAN weight (0.025).</summary>
    public double DiscriminatorWeight { get; set; } = 0.025;

    /// <summary>Gets or sets the decoder step the adversarial loss starts at (501).</summary>
    public int DiscriminatorStart { get; set; } = 501;

    #endregion
}
