using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the BasicVSR bidirectional recurrent video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Chan et al., CVPR 2021, and the authors' release (open-mmlab/mmagic, BasicVSRNet and
/// configs/basicvsr/basicvsr_2xb4_reds4.py): 64-channel features, 30 residual blocks per propagation branch, x4
/// upsampling, 15-frame training clips, Charbonnier loss, and Adam (0.9, 0.99) at 2e-4 with SPyNet at 0.125 times the
/// rate and frozen for the first 5,000 iterations, cosine-annealed to 1e-7 over 300,000 iterations.
/// </para>
/// <para>
/// <b>For Beginners:</b> BasicVSR treats video frames like a sequence. It processes frames
/// both forward in time and backward, so each frame benefits from information in both
/// directions. Optical flow tells the model how pixels moved between frames, so it can
/// align them before combining their features for higher resolution output.
/// </para>
/// </remarks>
public class BasicVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public BasicVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BasicVSROptions(BasicVSROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MidChannels = other.MidChannels;
        NumResBlocks = other.NumResBlocks;
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

    /// <summary>Gets or sets the feature channels of the propagation branches (64).</summary>
    public int MidChannels { get; set; } = 64;

    /// <summary>Gets or sets the residual blocks per propagation branch (30).</summary>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the frames of a training clip (15, the REDS recipe).</summary>
    public int NumFrames { get; set; } = 15;

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

    /// <summary>Gets or sets the iterations SPyNet stays frozen at the start of training (5,000).</summary>
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
