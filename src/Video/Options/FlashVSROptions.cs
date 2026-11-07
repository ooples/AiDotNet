using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for FlashVSR, real-time diffusion-based streaming video super-resolution.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Zhuang et al. 2025 (arXiv 2510.12747) and the release (OpenImagingLab/FlashVSR): Wan 2.1-T2V-1.3B as
/// the backbone (1,536 wide, 30 blocks of 12 heads, an 8,960-wide feed-forward, a 256-wide frequency embedding, 512 T5
/// tokens of 4,096) fine-tuned with LoRA of rank 384; the released LR projection (hidden widths 2,048 and 3,072, one DiT
/// layer); the block-sparse attention at sparse ratio 2, a three-chunk KV cache and an 11-block locality window; the tiny
/// conditional decoder (512, 256, 128, 128); and Wan 2.1's VAE (96 wide, multipliers 1, 2, 4, 4, two residual blocks) for
/// the training latents. Training is AdamW at 1e-5 with weight decay 0.01, LPIPS weighted 2.
/// </para>
/// <para>
/// <b>For Beginners:</b> FlashVSR sharpens video as it streams in. It turns a video generation model into a one-step
/// upscaler that only looks a few frames back and only at the parts of each frame that matter, which makes it fast enough
/// for real time.
/// </para>
/// </remarks>
public class FlashVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public FlashVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FlashVSROptions(FlashVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ModelDim = other.ModelDim;
        FfnDim = other.FfnDim;
        NumHeads = other.NumHeads;
        NumLayers = other.NumLayers;
        FrequencyDim = other.FrequencyDim;
        TextDim = other.TextDim;
        TextLength = other.TextLength;
        NormEpsilon = other.NormEpsilon;
        LoraRank = other.LoraRank;
        LqHiddenDim1 = other.LqHiddenDim1;
        LqHiddenDim2 = other.LqHiddenDim2;
        LqProjectionLayers = other.LqProjectionLayers;
        SparseRatio = other.SparseRatio;
        KvCacheChunks = other.KvCacheChunks;
        LocalRange = other.LocalRange;
        DecoderChannels = (int[])other.DecoderChannels.Clone();
        VaeDim = other.VaeDim;
        VaeDimMultipliers = (int[])other.VaeDimMultipliers.Clone();
        VaeResidualBlocks = other.VaeResidualBlocks;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        ColorFix = other.ColorFix;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        LpipsWeight = other.LpipsWeight;
        SigmaShift = other.SigmaShift;
        NumTrainTimesteps = other.NumTrainTimesteps;
        DecoderIterations = other.DecoderIterations;
        TeacherIterations = other.TeacherIterations;
        SparseIterations = other.SparseIterations;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>Gets or sets the DiT width (Wan 2.1-1.3B: 1,536).</summary>
    public int ModelDim { get; set; } = 1536;

    /// <summary>Gets or sets the feed-forward width (8,960).</summary>
    public int FfnDim { get; set; } = 8960;

    /// <summary>Gets or sets the attention heads (12). The head width must be even with a third that is even.</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the DiT blocks (30).</summary>
    public int NumLayers { get; set; } = 30;

    /// <summary>Gets or sets the sinusoidal timestep embedding width (256).</summary>
    public int FrequencyDim { get; set; } = 256;

    /// <summary>Gets or sets the T5 embedding width of the fixed prompt (4,096).</summary>
    public int TextDim { get; set; } = 4096;

    /// <summary>Gets or sets the fixed prompt's token count (512, Wan's text length).</summary>
    public int TextLength { get; set; } = 512;

    /// <summary>Gets or sets the norms' epsilon (1e-6).</summary>
    public double NormEpsilon { get; set; } = 1e-6;

    /// <summary>Gets or sets the LoRA rank on the DiT's attention and feed-forward maps (384, Sec. 4.1).</summary>
    public int LoraRank { get; set; } = 384;

    /// <summary>Gets or sets the LR projection's first hidden width (2,048).</summary>
    public int LqHiddenDim1 { get; set; } = 2048;

    /// <summary>Gets or sets the LR projection's second hidden width (3,072).</summary>
    public int LqHiddenDim2 { get; set; } = 3072;

    /// <summary>Gets or sets how many DiT blocks receive the LR tokens (the release's layer_num, 1).</summary>
    public int LqProjectionLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the sparse ratio (the release recommends 1.5 or 2; 2 is more stable). The top-k ratio is this times
    /// 768·1280 over the output area, so the kept share of blocks is the same at any resolution.
    /// </summary>
    public double SparseRatio { get; set; } = 2.0;

    /// <summary>Gets or sets how many earlier two-frame chunks the KV cache keeps (kv_ratio, 3).</summary>
    public int KvCacheChunks { get; set; } = 3;

    /// <summary>Gets or sets the locality window in 8x8-token blocks (the release recommends 9 or 11; 11 is more stable).</summary>
    public int LocalRange { get; set; } = 11;

    /// <summary>Gets or sets the tiny conditional decoder's stage widths (512, 256, 128, 128).</summary>
    public int[] DecoderChannels { get; set; } = { 512, 256, 128, 128 };

    /// <summary>Gets or sets Wan 2.1 VAE's base width (96).</summary>
    public int VaeDim { get; set; } = 96;

    /// <summary>Gets or sets the VAE's width multipliers (1, 2, 4, 4); four stages give its x8 and x4 compression.</summary>
    public int[] VaeDimMultipliers { get; set; } = { 1, 2, 4, 4 };

    /// <summary>Gets or sets the VAE's residual blocks a stage (2).</summary>
    public int VaeResidualBlocks { get; set; } = 2;

    #endregion

    #region Inference

    /// <summary>Gets or sets the upscaling factor (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the training clip length (89 frames, Sec. 4.1).</summary>
    public int NumFrames { get; set; } = 89;

    /// <summary>Gets or sets whether the output's per-frame color statistics are matched to the LR upscale (AdaIN, as the release does).</summary>
    public bool ColorFix { get; set; } = true;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the ONNX model path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the AdamW learning rate (1e-5).</summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>Gets or sets the AdamW weight decay (0.01).</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>Gets or sets the LPIPS weight λ in the distillation and decoder losses (2).</summary>
    public double LpipsWeight { get; set; } = 2.0;

    /// <summary>Gets or sets the flow-matching schedule's shift (Wan's 5).</summary>
    public double SigmaShift { get; set; } = 5.0;

    /// <summary>Gets or sets the flow-matching training timesteps (1,000).</summary>
    public int NumTrainTimesteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the tiny decoder's iterations. The paper gives each stage in days (decoder 2, then 2, 1 and 2 for the
    /// three DiT stages) and no iteration counts; the defaults keep those proportions.
    /// </summary>
    public int DecoderIterations { get; set; } = 20000;

    /// <summary>Gets or sets the full-attention teacher's flow-matching iterations (Stage 1).</summary>
    public int TeacherIterations { get; set; } = 20000;

    /// <summary>Gets or sets the block-sparse causal adaptation's iterations (Stage 2); distillation follows.</summary>
    public int SparseIterations { get; set; } = 10000;

    #endregion
}
