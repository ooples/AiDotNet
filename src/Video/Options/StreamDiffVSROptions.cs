using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the Stream-DiffVSR low-latency streaming video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// Stream-DiffVSR (Shiu et al., 2025, arXiv 2512.23709) achieves low-latency online video
/// super-resolution through:
/// - Auto-regressive temporal guidance (ARTG): warped previous HR output conditions current denoising
/// - 4-step distilled denoiser: rollout-distilled from a 50-step SD x4 Upscaler
/// - A Temporal Processor Module (TPM) after each spatial convolution in the decoder
/// - Causal temporal conditioning: only looks at past frames, enabling streaming applications
/// </para>
/// <para>
/// Defaults below are the paper's own training settings (appendix C), so a caller who configures
/// nothing reproduces the paper rather than an invented configuration. Every value remains settable.
/// </para>
/// <para>
/// <b>For Beginners:</b> Stream-DiffVSR is designed for live video upscaling where you can't
/// look at future frames. It uses a trick called "distillation" to reduce the number of
/// processing steps from ~50 to just 4, making it fast enough for real-time streaming.
/// </para>
/// </remarks>
public class StreamDiffVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public StreamDiffVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public StreamDiffVSROptions(StreamDiffVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDenoisingSteps = other.NumDenoisingSteps;
        NumResBlocks = other.NumResBlocks;
        TemporalRadius = other.TemporalRadius;
        ScaleFactor = other.ScaleFactor;
        LatentDim = other.LatentDim;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        WeightDecay = other.WeightDecay;
        LatentL2Weight = other.LatentL2Weight;
        DistillLpipsWeight = other.DistillLpipsWeight;
        DistillGanWeight = other.DistillGanWeight;
        ReconstructionWeight = other.ReconstructionWeight;
        DecoderLpipsWeight = other.DecoderLpipsWeight;
        FlowWeight = other.FlowWeight;
        DecoderGanWeight = other.DecoderGanWeight;
        AdversarialWarmupIterations = other.AdversarialWarmupIterations;
        DiscriminatorReceptiveField = other.DiscriminatorReceptiveField;
        DiscriminatorFilters = other.DiscriminatorFilters;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of denoising steps (distilled).</summary>
    /// <remarks>Original diffusion uses ~50 steps; distillation reduces to 4.</remarks>
    public int NumDenoisingSteps { get; set; } = 4;

    /// <summary>Gets or sets the number of residual blocks in the denoiser.</summary>
    public int NumResBlocks { get; set; } = 16;

    /// <summary>Gets or sets the temporal radius for causal conditioning (past frames only).</summary>
    public int TemporalRadius { get; set; } = 3;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the latent space dimension for the diffusion process.</summary>
    public int LatentDim { get; set; } = 64;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate. Default 5e-5 with a constant schedule, the value used for every
    /// training stage in appendix C.
    /// </summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the warmup steps for the learning rate schedule.</summary>
    public int WarmupSteps { get; set; } = 5000;

    /// <summary>Gets or sets AdamW's first moment decay. Default 0.9 (appendix C).</summary>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second moment decay. Default 0.999 (appendix C).</summary>
    public double AdamBeta2 { get; set; } = 0.999;

    /// <summary>Gets or sets AdamW's decoupled weight decay. Default 0.01 (appendix C).</summary>
    public double WeightDecay { get; set; } = 0.01;

    #endregion

    #region Objective Weights

    /// <summary>
    /// Gets or sets the weight on the latent MSE term of the rollout-distillation objective
    /// (<c>||z_den - z_gt||^2</c>). Default 1.0 (appendix C.1).
    /// </summary>
    /// <remarks>
    /// Rollout distillation applies its loss ONLY to the final denoised latent, not to random
    /// intermediate timesteps, so that the training trajectory mirrors inference (section 3.2).
    /// </remarks>
    public double LatentL2Weight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the perceptual (LPIPS) weight for the distillation objective. Default 0.5
    /// (appendix C.1).
    /// </summary>
    public double DistillLpipsWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the adversarial weight for the distillation objective. Default 0.025
    /// (appendix C.1).
    /// </summary>
    public double DistillGanWeight { get; set; } = 0.025;

    /// <summary>
    /// Gets or sets the SmoothL1 reconstruction weight for the temporal-decoder objective.
    /// Default 1.0 (appendix C.2).
    /// </summary>
    public double ReconstructionWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the perceptual (LPIPS) weight for the temporal-decoder objective. Default 0.3
    /// (appendix C.2).
    /// </summary>
    public double DecoderLpipsWeight { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the optical-flow consistency weight for the temporal-decoder objective.
    /// Default 0.1 (appendix C.2). The paper computes this term with RAFT.
    /// </summary>
    public double FlowWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the adversarial weight for the temporal-decoder objective. Default 0.025
    /// (appendix C.2).
    /// </summary>
    public double DecoderGanWeight { get; set; } = 0.025;

    /// <summary>
    /// Gets or sets the iteration after which the adversarial and flow terms are enabled.
    /// Default 20,000: "Flow loss and adversarial loss are envolved after 20k iteration for training
    /// stabilization" (appendix C.1/C.2).
    /// </summary>
    /// <remarks>
    /// Enabling a discriminator against an untrained generator destabilizes both, so the paper trains
    /// on reconstruction and perceptual terms alone first. Set to 0 to enable them immediately.
    /// </remarks>
    public int AdversarialWarmupIterations { get; set; } = 20_000;

    #endregion

    #region Discriminator

    /// <summary>
    /// Gets or sets the receptive field of the PatchGAN discriminator used for the adversarial terms.
    /// Default 70x70, the size pix2pix (Isola et al., 2017) reports as its best setting and the one
    /// Stream-DiffVSR cites.
    /// </summary>
    public PatchGANReceptiveField DiscriminatorReceptiveField { get; set; }
        = PatchGANReceptiveField.Patch70x70;

    /// <summary>
    /// Gets or sets the PatchGAN's first-block filter count. Default 64 (the "C64" of pix2pix
    /// section 6.1.2).
    /// </summary>
    public int DiscriminatorFilters { get; set; } = 64;

    #endregion
}
