using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.VFIformer{T}"/>, video frame interpolation with a
/// transformer (Lu et al., CVPR 2022).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's VFIformer (dvlab-research/VFIformer: models/archs/VFIformer_arch.py,
/// transformer_layers.py, models/trainer.py): IFNet's blocks of widths 240, 150 and 90, the width-24 flow refinement,
/// and the transformer of width 160 with 8x8 windows, two heads, MLP ratio 2 and two RTFL groups of 3, 3, 3 and 1 TFLs
/// at full, 1/2, 1/4 and 1/8 size, the second group at the first three sizes attending across scales. Training follows
/// the release: AdamW at 1e-4 with a weight decay of 1e-4, a 2,000-step linear warmup and a cosine to zero over 300
/// epochs, on the L1 plus ternary loss, plus 0.01 times the end-point error of IFNet's flows against precomputed flows
/// when they are given.
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIformer first estimates how pixels move between the two frames, warps both frames to the
/// middle, and then uses a transformer, which compares each small window of the image with its surroundings at a coarser
/// scale, to fix the places where the warping went wrong.
/// </para>
/// </remarks>
public class VFIformerOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public VFIformerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VFIformerOptions(VFIformerOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        FlowWidths = (int[])other.FlowWidths.Clone();
        RefineChannels = other.RefineChannels;
        EmbedDim = other.EmbedDim;
        WindowSize = other.WindowSize;
        NumHeads = other.NumHeads;
        MlpRatio = other.MlpRatio;
        Depths = (int[])other.Depths.Clone();
        CrossScaleAttention = other.CrossScaleAttention.Select(g => (bool[])g.Clone()).ToArray();
        DropPathRate = other.DropPathRate;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        TotalSteps = other.TotalSteps;
        L1Weight = other.L1Weight;
        TernaryWeight = other.TernaryWeight;
        FlowWeight = other.FlowWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>Gets or sets IFNet's three block widths, coarsest first (240, 150, 90); each is even.</summary>
    public int[] FlowWidths { get; set; } = { 240, 150, 90 };

    /// <summary>Gets or sets the flow refinement's base width c (24); the context runs from c to 8c.</summary>
    public int RefineChannels { get; set; } = 24;

    /// <summary>Gets or sets the transformer's width (embed_dim, 160).</summary>
    public int EmbedDim { get; set; } = 160;

    /// <summary>Gets or sets the attention window side (8), even and a multiple of 4; frame sides are padded to 8·window.</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the attention heads (2), dividing <see cref="EmbedDim"/>.</summary>
    public int NumHeads { get; set; } = 2;

    /// <summary>Gets or sets the MLP's hidden width as a multiple of <see cref="EmbedDim"/> (2).</summary>
    public double MlpRatio { get; set; } = 2.0;

    /// <summary>Gets or sets the TFLs in each of a scale's two RTFL groups, full size first (3, 3, 3, 1).</summary>
    public int[] Depths { get; set; } = { 3, 3, 3, 1 };

    /// <summary>
    /// Gets or sets, per scale (full size first) and per RTFL group, whether its TFLs also attend to the coarser windows:
    /// the second group at the first three scales.
    /// </summary>
    public bool[][] CrossScaleAttention { get; set; } =
    {
        new[] { false, true }, new[] { false, true }, new[] { false, true }, new[] { false, false },
    };

    /// <summary>Gets or sets the stochastic depth rate, rising linearly over each scale's TFLs (0.1).</summary>
    public double DropPathRate { get; set; } = 0.1;

    #endregion

    #region Training

    /// <summary>Gets or sets AdamW's peak learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets AdamW's decoupled weight decay (1e-4).</summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>Gets or sets the steps over which the rate rises linearly from zero (2,000).</summary>
    public int WarmupSteps { get; set; } = 2000;

    /// <summary>
    /// Gets or sets the step at which the cosine reaches zero: 300 epochs of Vimeo-90K's 51,312 triplets at the release's
    /// batch of 24 over 8 GPUs, 2,138 steps per epoch (641,400).
    /// </summary>
    public int TotalSteps { get; set; } = 641400;

    /// <summary>Gets or sets the L1 loss's weight (lambda_l1, 1).</summary>
    public double L1Weight { get; set; } = 1.0;

    /// <summary>Gets or sets the ternary census loss's weight (lambda_ter, 1).</summary>
    public double TernaryWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the flow end-point error's weight when flows are given (lambda_flow, 0.01).</summary>
    public double FlowWeight { get; set; } = 0.01;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
