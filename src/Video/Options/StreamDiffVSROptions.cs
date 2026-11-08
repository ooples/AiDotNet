using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Options for <see cref="Enhancement.StreamDiffVSR{T}"/>: the authors' released Stream-DiffVSR (jamichss/Stream-DiffVSR,
/// Jamichsu/Stream-DiffVSR on the Hugging Face Hub).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's configuration: the four-step distilled UNet in the Stable Diffusion x4 upscaler's layout
/// (widths 256, 512, 512, 1024, eight heads, 1024-wide text, <c>only_cross_attention</c> at the first three levels), the
/// Auto-regressive Temporal Guidance ControlNet copied from it (conditioning widths 64, 128, 256), the temporal tiny
/// autoencoder (TAESD-x4 with the Temporal Processor Module in its decoder), DDIM with v-prediction on the scaled-linear
/// schedule at trailing timesteps, RAFT-large flows with twelve updates, and the ARTG stage's training recipe
/// (<c>train/train_artg.sh</c>).
/// </para>
/// <para><b>For Beginners:</b> the defaults build the released model. Shrink the widths for experiments on modest
/// hardware; every value stays settable.</para>
/// </remarks>
public class StreamDiffVSROptions : ModelOptions
{
    /// <summary>Creates the released configuration.</summary>
    public StreamDiffVSROptions()
    {
    }

    /// <summary>Copies <paramref name="other"/>.</summary>
    public StreamDiffVSROptions(StreamDiffVSROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        BlockChannels = (int[])other.BlockChannels.Clone();
        LayersPerBlock = other.LayersPerBlock;
        AttentionHeads = other.AttentionHeads;
        OnlyCrossAttention = (bool[])other.OnlyCrossAttention.Clone();
        CrossAttentionDim = other.CrossAttentionDim;
        PromptLength = other.PromptLength;
        ConditioningChannels = (int[])other.ConditioningChannels.Clone();
        ConditioningScale = other.ConditioningScale;
        VaeChannels = other.VaeChannels;
        EncoderBlocks = (int[])other.EncoderBlocks.Clone();
        DecoderBlocks = (int[])other.DecoderBlocks.Clone();
        ScaleFactor = other.ScaleFactor;
        SamplingSteps = other.SamplingSteps;
        NumTrainTimesteps = other.NumTrainTimesteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        FlowIterations = other.FlowIterations;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
        WeightDecay = other.WeightDecay;
        MaxGradientNorm = other.MaxGradientNorm;
        WarmupSteps = other.WarmupSteps;
        LatentMseWeight = other.LatentMseWeight;
        LpipsWeight = other.LpipsWeight;
        GanWeight = other.GanWeight;
        GanStartIteration = other.GanStartIteration;
        DiscriminatorInterval = other.DiscriminatorInterval;
    }

    #region UNet and ARTG

    /// <summary>The UNet's level widths (released: 256, 512, 512, 1024); each splits into 32 groups.</summary>
    public int[] BlockChannels { get; set; } = { 256, 512, 512, 1024 };

    /// <summary>The residual blocks per encoder level; the decoder has one more (released: 2).</summary>
    public int LayersPerBlock { get; set; } = 2;

    /// <summary>The attention heads of every transformer (released <c>attention_head_dim</c>: 8).</summary>
    public int AttentionHeads { get; set; } = 8;

    /// <summary>Per level, whether both attentions read the text (released: true, true, true, false).</summary>
    public bool[] OnlyCrossAttention { get; set; } = { true, true, true, false };

    /// <summary>The text embedding's width (released: 1024, OpenCLIP ViT-H).</summary>
    public int CrossAttentionDim { get; set; } = 1024;

    /// <summary>The empty prompt's tokens (77).</summary>
    public int PromptLength { get; set; } = 77;

    /// <summary>The ARTG condition embedding's widths (released: 64, 128, 256).</summary>
    public int[] ConditioningChannels { get; set; } = { 64, 128, 256 };

    /// <summary>The scale on the ARTG residuals (released: 1).</summary>
    public double ConditioningScale { get; set; } = 1.0;

    #endregion

    #region Temporal autoencoder

    /// <summary>The tiny autoencoder's width (released: 64).</summary>
    public int VaeChannels { get; set; } = 64;

    /// <summary>The encoder's blocks per level (released: 1, 3, 3).</summary>
    public int[] EncoderBlocks { get; set; } = { 1, 3, 3 };

    /// <summary>The decoder's temporal blocks per level (released: 3, 3, 1).</summary>
    public int[] DecoderBlocks { get; set; } = { 3, 3, 1 };

    #endregion

    #region Sampling

    /// <summary>The upscaling factor, the autoencoder's compression (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>The DDIM steps at trailing timesteps (released: 4).</summary>
    public int SamplingSteps { get; set; } = 4;

    /// <summary>The training schedule's timesteps (1000).</summary>
    public int NumTrainTimesteps { get; set; } = 1000;

    /// <summary>β at the first timestep of the scaled-linear schedule (1e-4).</summary>
    public double BetaStart { get; set; } = 1e-4;

    /// <summary>β at the last timestep (0.02).</summary>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>RAFT's flow updates (torchvision's <c>raft_large</c>: 12).</summary>
    public int FlowIterations { get; set; } = 12;

    #endregion

    #region Model Loading

    /// <summary>The ONNX model's path in inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>AdamW's learning rate (released ARTG stage: 5e-5).</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>AdamW's β1 (0.9).</summary>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>AdamW's β2 (0.999).</summary>
    public double AdamBeta2 { get; set; } = 0.999;

    /// <summary>AdamW's ε (1e-8).</summary>
    public double AdamEpsilon { get; set; } = 1e-8;

    /// <summary>AdamW's weight decay (1e-2).</summary>
    public double WeightDecay { get; set; } = 1e-2;

    /// <summary>The gradient-norm clip (1); zero disables it.</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>The linear warmup before the constant rate (<c>constant_with_warmup</c>, 1000 steps).</summary>
    public int WarmupSteps { get; set; } = 1000;

    /// <summary>The weight of the latent v-prediction MSE (1).</summary>
    public double LatentMseWeight { get; set; } = 1.0;

    /// <summary>The weight of LPIPS on the decoded frame (0.1); zero leaves AlexNet out.</summary>
    public double LpipsWeight { get; set; } = 0.1;

    /// <summary>The adversarial weight (0.025); zero leaves the discriminator out.</summary>
    public double GanWeight { get; set; } = 0.025;

    /// <summary>
    /// The step from which the adversarial term counts (the paper's 20k; the release's ARTG script passes 2000).
    /// </summary>
    public int GanStartIteration { get; set; } = 20_000;

    /// <summary>The discriminator trains, and the adversarial term applies, on every this many steps (5).</summary>
    public int DiscriminatorInterval { get; set; } = 5;

    #endregion
}
