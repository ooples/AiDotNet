using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.M2M{T}"/>, many-to-many splatting for video
/// frame interpolation (Hu, Niklaus, Sclaroff and Saenko, CVPR 2022).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's M2M-PWC (feinanshan/M2M_VFI: Train/vfi/model/m2m.py, pwcnet.py, configs/vimeo90k.yml):
/// PWC-Net flows at a quarter of the input size, a Motion Refinement Network of base width 16 that predicts four flows
/// per pixel each way, and a rank-16 attention cube. Training follows the release: AdamW at 1e-4 with a weight decay of
/// 1e-4, a 2,000-step linear warmup and cosine decay to zero over 900,000 steps, gradient norm clipped to 1, the flow
/// estimator held at a learning rate of zero, on the Charbonnier plus ternary census loss.
/// </para>
/// <para>
/// <b>For Beginners:</b> M2M estimates rough motion once, then refines it into several motion guesses for every pixel.
/// Each pixel is pushed forward along all of its guesses to the in-between time, and where several pixels land on the
/// same spot they are blended by how reliable each one is. There is no heavy synthesis network, so it is fast.
/// </para>
/// </remarks>
public class M2MOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public M2MOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public M2MOptions(M2MOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        BaseChannels = other.BaseChannels;
        NumBranches = other.NumBranches;
        CubeRank = other.CubeRank;
        Ratio = other.Ratio;
        FlowChannels = other.FlowChannels;
        FlowDecoderChannels = (int[])other.FlowDecoderChannels.Clone();
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        TotalSteps = other.TotalSteps;
        MaxGradientNorm = other.MaxGradientNorm;
        FlowLearningRateMultiplier = other.FlowLearningRateMultiplier;
        CharbonnierWeight = other.CharbonnierWeight;
        TernaryWeight = other.TernaryWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>Gets or sets the Motion Refinement Network's base width c (16); its widths run from c to 16c.</summary>
    public int BaseChannels { get; set; } = 16;

    /// <summary>Gets or sets the flows predicted per pixel in each direction (branch, 4).</summary>
    public int NumBranches { get; set; } = 4;

    /// <summary>Gets or sets the rank of the channel-height-width attention cube at 1/16 size (16).</summary>
    public int CubeRank { get; set; } = 16;

    /// <summary>Gets or sets the reduction, a power of two of at least 2, at which PWC-Net estimates the coarse flows (ratio, 4).</summary>
    public int Ratio { get; set; } = 4;

    /// <summary>Gets or sets PWC-Net's feature width at every pyramid level (32).</summary>
    public int FlowChannels { get; set; } = 32;

    /// <summary>Gets or sets the widths of each PWC-Net decoder's five hidden convolutions (128, 128, 96, 64, 32).</summary>
    public int[] FlowDecoderChannels { get; set; } = { 128, 128, 96, 64, 32 };

    #endregion

    #region Training

    /// <summary>Gets or sets AdamW's peak learning rate (lr0, 1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets AdamW's decoupled weight decay (wd, 1e-4).</summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>Gets or sets the steps over which the learning rate rises linearly from zero (warmup_steps, 2,000).</summary>
    public int WarmupSteps { get; set; } = 2000;

    /// <summary>Gets or sets the step at which the cosine decay reaches zero (max_iter, 900,000).</summary>
    public int TotalSteps { get; set; } = 900000;

    /// <summary>Gets or sets the gradient norm clip (1).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the flow estimator's learning rate as a multiple of the network's (0: the release loads a pretrained
    /// PWC-Net and holds it fixed). Set it above zero to train the flow estimator from scratch.
    /// </summary>
    public double FlowLearningRateMultiplier { get; set; } = 0.0;

    /// <summary>Gets or sets the weight of the Charbonnier loss (1).</summary>
    public double CharbonnierWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the ternary census loss (1).</summary>
    public double TernaryWeight { get; set; } = 1.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
