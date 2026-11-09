using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for TTVSR, the trajectory-aware transformer for video super-resolution.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Liu et al., CVPR 2022, and the authors' release (researchmm/TTVSR, TTVSRNet and
/// configs/TTVSR_reds4.py): 64-channel features, 60 residual blocks in the shared propagation trunk, 4x4 tokens,
/// keyframes every third frame, x4 upsampling, 50-frame training clips, Charbonnier loss, and Adam (0.9, 0.99) at 2e-4
/// with SPyNet at 0.125 times the rate and frozen for the first 5,000 iterations, cosine-annealed to 1e-7 over 400,000
/// iterations.
/// </para>
/// <para>
/// <b>For Beginners:</b> TTVSR follows where each patch of the picture moves through the video, and
/// borrows detail from the same patch in earlier frames along that path, even many frames back.
/// </para>
/// </remarks>
public class TTVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TTVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TTVSROptions(TTVSROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MidChannels = other.MidChannels;
        NumResBlocks = other.NumResBlocks;
        NumFrames = other.NumFrames;
        TokenStride = other.TokenStride;
        KeyframeStride = other.KeyframeStride;
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

    /// <summary>Gets or sets the residual blocks of the propagation trunk both directions share (60).</summary>
    public int NumResBlocks { get; set; } = 60;

    /// <summary>Gets or sets the frames of a training clip (50, the REDS recipe).</summary>
    public int NumFrames { get; set; } = 50;

    /// <summary>
    /// Gets or sets the token size (4): frames are split into TokenStride x TokenStride tokens, and their sizes must be
    /// multiples of it.
    /// </summary>
    public int TokenStride { get; set; } = 4;

    /// <summary>Gets or sets the keyframe spacing whose tokens later frames attend to (3).</summary>
    public int KeyframeStride { get; set; } = 3;

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

    /// <summary>Gets or sets the iterations of the cosine schedule (400,000).</summary>
    public int CosineIterations { get; set; } = 400000;

    /// <summary>
    /// Gets or sets the Charbonnier ε; the release's loss is sqrt(d² + 1e-12), so ε = 1e-6.
    /// </summary>
    public double CharbonnierEpsilon { get; set; } = 1e-6;

    #endregion
}
