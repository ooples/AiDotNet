using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Options for <see cref="Enhancement.SeedVR{T}"/>: the authors' released SeedVR-7B (ByteDance-Seed/SeedVR,
/// <c>configs_7b/main.yaml</c>), a window-attention MM-DiT on a causal video VAE.
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's configuration: the 36-block NaDiT at width 3072 with 24 heads of 128, windows counted at
/// 720p as (4, 3, 3) alternating with their half-shifted copies, the 16-channel VAE with 8x spatial and 4x temporal
/// compression, the precomputed positive and negative prompt embeddings (58 and 64 tokens of 5120), classifier-free guidance
/// and 50 Euler steps of the rectified flow. The paper's own experiments use a 2.48B model of the same design initialized from
/// SD3-Medium; set <see cref="ModelDim"/>, <see cref="NumHeads"/>, <see cref="HeadDim"/> and <see cref="NumLayers"/> for it.
/// </para>
/// <para><b>For Beginners:</b> the defaults build the released 7-billion-parameter model. Shrink the widths and depth for
/// experiments on modest hardware.</para>
/// </remarks>
public class SeedVROptions : NeuralNetworkOptions
{
    /// <summary>Creates the released configuration.</summary>
    public SeedVROptions()
    {
    }

    /// <summary>Copies <paramref name="other"/>.</summary>
    public SeedVROptions(SeedVROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ModelDim = other.ModelDim;
        NumHeads = other.NumHeads;
        HeadDim = other.HeadDim;
        ExpandRatio = other.ExpandRatio;
        NumLayers = other.NumLayers;
        NormEpsilon = other.NormEpsilon;
        QkBias = other.QkBias;
        WindowFrames = other.WindowFrames;
        WindowRows = other.WindowRows;
        WindowColumns = other.WindowColumns;
        TextDim = other.TextDim;
        TextLength = other.TextLength;
        NegativeTextLength = other.NegativeTextLength;
        VaeChannels = (int[])other.VaeChannels.Clone();
        VaeLayersPerBlock = other.VaeLayersPerBlock;
        VaeGroups = other.VaeGroups;
        LatentChannels = other.LatentChannels;
        LatentScaling = other.LatentScaling;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        SamplingSteps = other.SamplingSteps;
        GuidanceScale = other.GuidanceScale;
        GuidanceRescale = other.GuidanceRescale;
        ConditionNoiseScale = other.ConditionNoiseScale;
        TrainingConditionNoiseScale = other.TrainingConditionNoiseScale;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    #region NaDiT

    /// <summary>The DiT width, <c>vid_dim</c> = <c>txt_dim</c> (released 7B: 3072).</summary>
    public int ModelDim { get; set; } = 3072;

    /// <summary>The attention heads (released 7B: 24).</summary>
    public int NumHeads { get; set; } = 24;

    /// <summary>The head width (released: 128).</summary>
    public int HeadDim { get; set; } = 128;

    /// <summary>The MLP's expansion (released: 4).</summary>
    public int ExpandRatio { get; set; } = 4;

    /// <summary>The blocks (released 7B: 36).</summary>
    public int NumLayers { get; set; } = 36;

    /// <summary>The RMS norms' ε (released: 1e-5).</summary>
    public double NormEpsilon { get; set; } = 1e-5;

    /// <summary>Whether the q, k, v projections have biases (released: false).</summary>
    public bool QkBias { get; set; }

    /// <summary>The windows along time at 720p (released: 4).</summary>
    public int WindowFrames { get; set; } = 4;

    /// <summary>The windows down the height at 720p (released: 3).</summary>
    public int WindowRows { get; set; } = 3;

    /// <summary>The windows across the width at 720p (released: 3).</summary>
    public int WindowColumns { get; set; } = 3;

    /// <summary>The prompt embeddings' width (released: 5120).</summary>
    public int TextDim { get; set; } = 5120;

    /// <summary>The positive prompt's tokens (the released <c>pos_emb.pt</c>: 58).</summary>
    public int TextLength { get; set; } = 58;

    /// <summary>The negative prompt's tokens (the released <c>neg_emb.pt</c>: 64).</summary>
    public int NegativeTextLength { get; set; } = 64;

    #endregion

    #region VAE

    /// <summary>The VAE's four stage widths (released: 128, 256, 512, 512).</summary>
    public int[] VaeChannels { get; set; } = { 128, 256, 512, 512 };

    /// <summary>The encoder's residual blocks per stage; the decoder has one more (released: 2).</summary>
    public int VaeLayersPerBlock { get; set; } = 2;

    /// <summary>The VAE's group-norm groups (released: 32).</summary>
    public int VaeGroups { get; set; } = 32;

    /// <summary>The latent channels (released: 16).</summary>
    public int LatentChannels { get; set; } = 16;

    /// <summary>The latent scaling factor (released: 0.9152).</summary>
    public double LatentScaling { get; set; } = 0.9152;

    #endregion

    #region Video and sampling

    /// <summary>The upscaling factor (4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>The frames of a clip.</summary>
    public int NumFrames { get; set; } = 21;

    /// <summary>The Euler steps over the uniform trailing timesteps (released: 50).</summary>
    public int SamplingSteps { get; set; } = 50;

    /// <summary>The classifier-free guidance scale (the release's inference script: 6.5); 1 runs the positive prompt only.</summary>
    public double GuidanceScale { get; set; } = 6.5;

    /// <summary>The guidance rescale (released: 0).</summary>
    public double GuidanceRescale { get; set; }

    /// <summary>The LR condition's noise augmentation at inference, as a fraction of the schedule (the release's script: 0.1).</summary>
    public double ConditionNoiseScale { get; set; } = 0.1;

    /// <summary>The LR condition's noise augmentation in training (released config: 0.25).</summary>
    public double TrainingConditionNoiseScale { get; set; } = 0.25;

    #endregion

    #region Model Loading

    /// <summary>The ONNX model's path in inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>AdamW's learning rate (SD3's 1e-4; the paper follows SD3's training settings).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>AdamW's weight decay (SD3 states none; 0).</summary>
    public double WeightDecay { get; set; }

    #endregion
}
