using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for DOVE, efficient one-step diffusion for real-world video super-resolution.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Chen et al., NeurIPS 2025 (arXiv 2505.16239) and the release (zhengchen1999/DOVE), which fine-tunes
/// CogVideoX1.5-5B as diffusers ships it (<c>THUDM/CogVideoX1.5-5B</c>): the 42-layer expert transformer (48 heads of
/// 64, 2x2x2 patches, a 512-wide time embedding, 4,096-wide T5 text embeddings of 226 tokens) and the causal 3D VAE
/// (128, 256, 256, 512 channels, 16 latent channels, x8 in space and x4 in time, latent scale 0.7). The x4 bilinear
/// upscale of the clip is encoded and denoised in one v-prediction step at timestep 399 of the zero-terminal-SNR
/// scaled-linear schedule, conditioned on the pre-encoded empty prompt. Training is AdamW (β 0.9, 0.95, weight decay 1e-4, gradients clipped
/// to 0.1, a 100-step linear warmup): 10,000
/// latent-space MSE iterations at 2e-5, then 500 pixel-space iterations at 5e-6 on MSE, DISTS and the frame-difference
/// loss (1, 1), 80% of them on single frames.
/// </para>
/// <para>
/// <b>For Beginners:</b> DOVE upscales video by asking a large video generation model to clean up a blurry enlargement
/// of it in a single step, which makes it far faster than diffusion upscalers that take dozens of steps.
/// </para>
/// </remarks>
public class DOVEOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public DOVEOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DOVEOptions(DOVEOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        NumAttentionHeads = other.NumAttentionHeads;
        AttentionHeadDim = other.AttentionHeadDim;
        NumLayers = other.NumLayers;
        TimeEmbedDim = other.TimeEmbedDim;
        TextEmbedDim = other.TextEmbedDim;
        MaxTextSeqLength = other.MaxTextSeqLength;
        PatchSize = other.PatchSize;
        PatchSizeT = other.PatchSizeT;
        NormEpsilon = other.NormEpsilon;
        VaeChannels = (int[])other.VaeChannels.Clone();
        VaeLayersPerBlock = other.VaeLayersPerBlock;
        LatentChannels = other.LatentChannels;
        TemporalCompressionRatio = other.TemporalCompressionRatio;
        LatentScale = other.LatentScale;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        Timestep = other.Timestep;
        NumTrainTimesteps = other.NumTrainTimesteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        RefinementLearningRate = other.RefinementLearningRate;
        WeightDecay = other.WeightDecay;
        MaxGradientNorm = other.MaxGradientNorm;
        WarmupSteps = other.WarmupSteps;
        AdaptationIterations = other.AdaptationIterations;
        DistsWeight = other.DistsWeight;
        FrameDifferenceWeight = other.FrameDifferenceWeight;
        ImageRatio = other.ImageRatio;
    }

    #region Transformer

    /// <summary>Gets or sets the transformer's attention heads (48).</summary>
    public int NumAttentionHeads { get; set; } = 48;

    /// <summary>Gets or sets each head's width (64); a multiple of 16 so the rotary embedding splits into pairs.</summary>
    public int AttentionHeadDim { get; set; } = 64;

    /// <summary>Gets or sets the transformer blocks (42).</summary>
    public int NumLayers { get; set; } = 42;

    /// <summary>Gets or sets the time embedding's width (512).</summary>
    public int TimeEmbedDim { get; set; } = 512;

    /// <summary>Gets or sets the text embeddings' width (4,096, T5-XXL's).</summary>
    public int TextEmbedDim { get; set; } = 4096;

    /// <summary>Gets or sets the text tokens (226).</summary>
    public int MaxTextSeqLength { get; set; } = 226;

    /// <summary>Gets or sets the spatial patch side (2).</summary>
    public int PatchSize { get; set; } = 2;

    /// <summary>Gets or sets the temporal patch length (2, CogVideoX 1.5's).</summary>
    public int PatchSizeT { get; set; } = 2;

    /// <summary>Gets or sets the transformer's LayerNorm epsilon (1e-5).</summary>
    public double NormEpsilon { get; set; } = 1e-5;

    #endregion

    #region VAE

    /// <summary>Gets or sets the causal VAE's stage widths (128, 256, 256, 512); each a multiple of 32.</summary>
    public int[] VaeChannels { get; set; } = { 128, 256, 256, 512 };

    /// <summary>Gets or sets the VAE encoder's residual blocks per stage (3; the decoder has one more).</summary>
    public int VaeLayersPerBlock { get; set; } = 3;

    /// <summary>Gets or sets the latent channels (16).</summary>
    public int LatentChannels { get; set; } = 16;

    /// <summary>Gets or sets the VAE's temporal compression (4, a power of two).</summary>
    public int TemporalCompressionRatio { get; set; } = 4;

    /// <summary>Gets or sets the latent scale (0.7).</summary>
    public double LatentScale { get; set; } = 0.7;

    #endregion

    #region Super-resolution and schedule

    /// <summary>Gets or sets the spatial upscaling factor (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the frames of a clip (25, the paper's training length).</summary>
    public int NumFrames { get; set; } = 25;

    /// <summary>Gets or sets the timestep the one denoising step runs at (399).</summary>
    public int Timestep { get; set; } = 399;

    /// <summary>Gets or sets the schedule's training timesteps (1,000).</summary>
    public int NumTrainTimesteps { get; set; } = 1000;

    /// <summary>Gets or sets the scaled-linear schedule's first β (0.00085).</summary>
    public double BetaStart { get; set; } = 0.00085;

    /// <summary>Gets or sets the scaled-linear schedule's last β (0.012).</summary>
    public double BetaEnd { get; set; } = 0.012;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the adaptation stage's AdamW rate (2e-5).</summary>
    public double LearningRate { get; set; } = 2e-5;

    /// <summary>Gets or sets the refinement stage's AdamW rate (5e-6).</summary>
    public double RefinementLearningRate { get; set; } = 5e-6;

    /// <summary>Gets or sets AdamW's decoupled weight decay (1e-4, the release's).</summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>Gets or sets the gradient norm clip (0.1, the release's training scripts).</summary>
    public double MaxGradientNorm { get; set; } = 0.1;

    /// <summary>Gets or sets the linear warmup steps of each stage's otherwise constant rate (100, the release's).</summary>
    public int WarmupSteps { get; set; } = 100;

    /// <summary>Gets or sets the latent-space adaptation iterations before pixel-space refinement (10,000).</summary>
    public int AdaptationIterations { get; set; } = 10000;

    /// <summary>Gets or sets the refinement stage's DISTS weight λ1 (1; 0 leaves the VGG16 out).</summary>
    public double DistsWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the refinement stage's frame-difference weight λ2 (1).</summary>
    public double FrameDifferenceWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the share of refinement iterations trained on a single frame (0.8).</summary>
    public double ImageRatio { get; set; } = 0.8;

    #endregion
}
