using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for RVRT, the recurrent video restoration transformer with guided deformable attention.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Liang et al., NeurIPS 2022, and the authors' release (JingyunLiang/RVRT network_rvrt.py and KAIR
/// options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json): 144-channel embeddings, 6 attention heads, 2x8x8 windows,
/// RSTB groups of depth 2 ([1, 2, 1] blocks for extraction, each propagation pass and reconstruction), guided deformable
/// attention with 12 groups and 12 heads over 3x3 windows and offset residues up to 10 pixels, two-frame clips, 30-frame
/// training sequences, Charbonnier loss (eps 1e-9 under the square root), and Adam (0.9, 0.99) at 4e-4 with SPyNet at
/// 0.25 times the rate and frozen for the first 30,000 iterations, cosine-annealed to 1e-7 over 600,000 iterations.
/// </para>
/// <para>
/// <b>For Beginners:</b> RVRT splits a video into pairs of frames and walks through them backward and forward in time,
/// carrying what it learned from earlier pairs. To line up a pair with the previous one it follows the estimated
/// motion and then looks around each moved point for the best-matching detail.
/// </para>
/// </remarks>
public class RVRTOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RVRTOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RVRTOptions(RVRTOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EmbedDim = other.EmbedDim;
        NumHeads = other.NumHeads;
        WindowSize = other.WindowSize;
        Depth = other.Depth;
        ExtractionBlocks = other.ExtractionBlocks;
        PropagationBlocks = other.PropagationBlocks;
        ReconstructionBlocks = other.ReconstructionBlocks;
        MlpRatio = other.MlpRatio;
        DeformableGroups = other.DeformableGroups;
        AttentionHeads = other.AttentionHeads;
        MaxResidueMagnitude = other.MaxResidueMagnitude;
        NumFrames = other.NumFrames;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        SpyNetLearningRateMultiplier = other.SpyNetLearningRateMultiplier;
        FlowFreezeIterations = other.FlowFreezeIterations;
        MinLearningRate = other.MinLearningRate;
        CosineIterations = other.CosineIterations;
        CharbonnierEpsilon = other.CharbonnierEpsilon;
    }

    #region Architecture

    /// <summary>Gets or sets the embedding width of every stage (144).</summary>
    public int EmbedDim { get; set; } = 144;

    /// <summary>Gets or sets the window-attention heads (6).</summary>
    public int NumHeads { get; set; } = 6;

    /// <summary>Gets or sets the spatial window size (8; windows are 2x8x8 in propagation, 1x8x8 elsewhere).</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the Swin layers per residual block (2).</summary>
    public int Depth { get; set; } = 2;

    /// <summary>Gets or sets the residual Swin blocks of the feature extractor (1).</summary>
    public int ExtractionBlocks { get; set; } = 1;

    /// <summary>Gets or sets the residual Swin blocks of each propagation pass (2).</summary>
    public int PropagationBlocks { get; set; } = 2;

    /// <summary>Gets or sets the residual Swin blocks of the reconstruction (1).</summary>
    public int ReconstructionBlocks { get; set; } = 1;

    /// <summary>Gets or sets the MLP width as a multiple of the embedding (2).</summary>
    public double MlpRatio { get; set; } = 2.0;

    /// <summary>Gets or sets the deformable offset groups of the guided deformable attention (12).</summary>
    public int DeformableGroups { get; set; } = 12;

    /// <summary>Gets or sets the heads of the guided deformable attention (12).</summary>
    public int AttentionHeads { get; set; } = 12;

    /// <summary>Gets or sets the largest offset residue on top of the flow, in pixels (10).</summary>
    public double MaxResidueMagnitude { get; set; } = 10;

    /// <summary>Gets or sets the frames of a training sequence (30, the REDS recipe; a multiple of the clip size 2).</summary>
    public int NumFrames { get; set; } = 30;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's starting learning rate (4e-4).</summary>
    public double LearningRate { get; set; } = 4e-4;

    /// <summary>Gets or sets the multiplier of SPyNet's learning rate (0.25, i.e. 1e-4).</summary>
    public double SpyNetLearningRateMultiplier { get; set; } = 0.25;

    /// <summary>Gets or sets the iterations SPyNet stays frozen at the start of training (30,000).</summary>
    public int FlowFreezeIterations { get; set; } = 30000;

    /// <summary>Gets or sets the floor of the cosine schedule (1e-7).</summary>
    public double MinLearningRate { get; set; } = 1e-7;

    /// <summary>Gets or sets the iterations of the cosine schedule (600,000).</summary>
    public int CosineIterations { get; set; } = 600000;

    /// <summary>
    /// Gets or sets the Charbonnier ε; KAIR's loss is sqrt(d² + 1e-9), so ε = sqrt(1e-9).
    /// </summary>
    public double CharbonnierEpsilon { get; set; } = 3.1622776601683795e-5;

    #endregion
}
