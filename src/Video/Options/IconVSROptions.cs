using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for IconVSR, BasicVSR with information refill and coupled propagation.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Chan et al., CVPR 2021, and the authors' release (open-mmlab/mmagic, IconVSRNet and
/// configs/iconvsr/iconvsr_2xb4_reds4.py): 64-channel features, 30 residual blocks per propagation branch, keyframes
/// every 5 frames refilled from an EDVR-M feature extractor over 5 frames (64 channels, 8 deformable groups, 5
/// extraction blocks), x4 upsampling, 15-frame training clips, Charbonnier loss, and BasicVSR's schedule: Adam
/// (0.9, 0.99) at 2e-4 with SPyNet at 0.125 times the rate, SPyNet and EDVR frozen for the first 5,000 iterations,
/// cosine-annealed to 1e-7 over 300,000 iterations.
/// </para>
/// <para>
/// <b>For Beginners:</b> In long videos, processing frames one-by-one can accumulate errors.
/// IconVSR picks "keyframes" as reliable reference points, re-injecting features computed directly
/// from the neighboring frames there, and lets the forward pass see what the backward pass found.
/// </para>
/// </remarks>
public class IconVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public IconVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public IconVSROptions(IconVSROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MidChannels = other.MidChannels;
        NumResBlocks = other.NumResBlocks;
        NumFrames = other.NumFrames;
        KeyframeStride = other.KeyframeStride;
        RefillPadding = other.RefillPadding;
        EdvrChannels = other.EdvrChannels;
        DeformableGroups = other.DeformableGroups;
        EdvrExtractionBlocks = other.EdvrExtractionBlocks;
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

    /// <summary>Gets or sets the feature channels of the propagation branches (64).</summary>
    public int MidChannels { get; set; } = 64;

    /// <summary>Gets or sets the residual blocks per propagation branch (30).</summary>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the frames of a training clip (15, the REDS recipe).</summary>
    public int NumFrames { get; set; } = 15;

    /// <summary>Gets or sets the keyframe spacing: frames 0, s, 2s, ... and the last are refilled (5).</summary>
    public int KeyframeStride { get; set; } = 5;

    /// <summary>
    /// Gets or sets the frames on each side of a keyframe the refill extractor sees (2 for REDS, 3 for Vimeo-90K).
    /// </summary>
    public int RefillPadding { get; set; } = 2;

    /// <summary>Gets or sets the EDVR-M feature extractor's channels (64).</summary>
    public int EdvrChannels { get; set; } = 64;

    /// <summary>Gets or sets the deformable groups of EDVR's PCD alignment (8).</summary>
    public int DeformableGroups { get; set; } = 8;

    /// <summary>Gets or sets the EDVR extractor's per-frame residual blocks (5).</summary>
    public int EdvrExtractionBlocks { get; set; } = 5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's starting learning rate (2e-4).</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the multiplier of SPyNet's learning rate (0.125, i.e. 2.5e-5).</summary>
    public double SpyNetLearningRateMultiplier { get; set; } = 0.125;

    /// <summary>Gets or sets the iterations SPyNet and the EDVR extractor stay frozen at the start of training (5,000).</summary>
    public int FlowFreezeIterations { get; set; } = 5000;

    /// <summary>Gets or sets the floor of the cosine schedule (1e-7).</summary>
    public double MinLearningRate { get; set; } = 1e-7;

    /// <summary>Gets or sets the iterations of the cosine schedule (300,000).</summary>
    public int CosineIterations { get; set; } = 300000;

    /// <summary>
    /// Gets or sets the Charbonnier ε; the release's loss is sqrt(d² + 1e-12), so ε = 1e-6.
    /// </summary>
    public double CharbonnierEpsilon { get; set; } = 1e-6;

    #endregion
}
