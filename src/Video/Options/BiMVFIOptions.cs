using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.BiMVFI{T}"/>, bidirectional motion
/// field-guided frame interpolation (Seo, Oh and Kim, CVPR 2025).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's (KAIST-VICLab/BiM-VFI, cfgs/bim_vfi.yaml and modules/components/bim_vfi): 32 base
/// channels, three pyramid levels, AdamW under a one-cycle schedule with gradient clipping and the six-term loss.
/// </para>
/// <para>
/// <b>For Beginners:</b> BiM-VFI describes how each pixel moves between the frames by a speed ratio and an angle
/// (the bidirectional motion field), estimates the two motions from the in-between time back to each frame
/// coarse-to-fine, and blends the two warped frames into the in-between frame.
/// </para>
/// </remarks>
public class BiMVFIOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public BiMVFIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BiMVFIOptions(BiMVFIOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        FeatureChannels = other.FeatureChannels;
        PyramidLevels = other.PyramidLevels;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        TotalIterations = other.TotalIterations;
        WarmupFraction = other.WarmupFraction;
        MaxGradientNorm = other.MaxGradientNorm;
        LevelDecay = other.LevelDecay;
        FlowLevelDecay = other.FlowLevelDecay;
        DistillationWeight = other.DistillationWeight;
        RegularizationWeight = other.RegularizationWeight;
        SmoothnessWeight = other.SmoothnessWeight;
        SmoothnessEdgeConstant = other.SmoothnessEdgeConstant;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>Gets or sets the base feature width; the encoders reach four times it (32).</summary>
    public int FeatureChannels { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of image pyramid levels (3, the release's training value; the paper runs 5 on
    /// SNU-FILM and 7 on XTest). Frames are padded to multiples of 2^(levels + 1).
    /// </summary>
    public int PyramidLevels { get; set; } = 3;

    #endregion

    #region Training

    /// <summary>Gets or sets the one-cycle schedule's peak AdamW learning rate (1.5e-4).</summary>
    public double LearningRate { get; set; } = 1.5e-4;

    /// <summary>Gets or sets AdamW's weight decay (1e-4).</summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the one-cycle schedule's length: 400 epochs of Vimeo90K's 64,612 septuplets at the paper's
    /// batch of 32.
    /// </summary>
    public int TotalIterations { get; set; } = 400 * (64612 / 32);

    /// <summary>Gets or sets the share of the schedule spent warming up to the peak rate (0.01).</summary>
    public double WarmupFraction { get; set; } = 0.01;

    /// <summary>Gets or sets the gradient-norm clip (1).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>Gets or sets the per-level weight of the photometric losses, coarser levels weighted less (0.25).</summary>
    public double LevelDecay { get; set; } = 0.25;

    /// <summary>Gets or sets the per-level weight of the flow losses (0.5).</summary>
    public double FlowLevelDecay { get; set; } = 0.5;

    /// <summary>Gets or sets the weight of the student-to-teacher flow distillation (0.005).</summary>
    public double DistillationWeight { get; set; } = 0.005;

    /// <summary>Gets or sets the weight of the teacher's zero-flow regularization (0.005).</summary>
    public double RegularizationWeight { get; set; } = 0.005;

    /// <summary>Gets or sets the weight of the teacher flows' edge-aware smoothness (0.05).</summary>
    public double SmoothnessWeight { get; set; } = 0.05;

    /// <summary>Gets or sets the smoothness term's edge constant (150).</summary>
    public double SmoothnessEdgeConstant { get; set; } = 150.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
