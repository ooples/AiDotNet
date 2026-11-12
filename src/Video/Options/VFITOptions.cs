using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.VFIT{T}"/>, the video frame interpolation
/// transformer (Shi et al., CVPR 2022).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's VFIT-B (zhshi0816/Video-Frame-Interpolation-Transformer: model/VFIT_B.py,
/// Sep_STS_Encoder.py, Sep_STS_Layer.py, main.py, config.py): four input frames, stage widths 64, 128, 256 and 512 with
/// 2, 2, 6 and 2 Sep-STS blocks and 2, 4, 8 and 16 heads, 8x8 spatial windows, an MLP ratio of 4, and 5x5 AdaCoF kernels
/// from 64-channel features. VFIT-S is the same with the widths 32, 64, 128 and 192
/// (<see cref="Widths"/> = { 192, 128, 64, 32 }). Training follows the release: Adamax (betas 0.9 and 0.999) on the L1
/// loss, its rate stepped through 2e-4, 1e-4, 5e-5, 2.5e-5, 5e-6 and 1e-6 at epochs 40, 60, 75, 85 and 95 of 100.
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIT looks at two frames before and two after the missing one. A transformer compares small
/// windows within each frame and the same pixel across the four frames, and the result decides, for every output pixel,
/// which nearby input pixels to blend and by how much.
/// </para>
/// </remarks>
public class VFITOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public VFITOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VFITOptions(VFITOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Widths = (int[])other.Widths.Clone();
        NumInputFrames = other.NumInputFrames;
        StageDepths = (int[])other.StageDepths.Clone();
        StageHeads = (int[])other.StageHeads.Clone();
        WindowSize = other.WindowSize;
        MlpRatio = other.MlpRatio;
        SynthesisChannels = other.SynthesisChannels;
        KernelSize = other.KernelSize;
        Dilation = other.Dilation;
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        LearningRateMilestones = (int[])other.LearningRateMilestones.Clone();
        LearningRateSteps = (double[])other.LearningRateSteps.Clone();
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>
    /// Gets or sets the stage widths, deepest first (nf: 512, 256, 128, 64 for VFIT-B; 192, 128, 64, 32 for VFIT-S); the
    /// stem is half the last.
    /// </summary>
    public int[] Widths { get; set; } = { 512, 256, 128, 64 };

    /// <summary>Gets or sets the input frames, two before and two after the target (4).</summary>
    public int NumInputFrames { get; set; } = 4;

    /// <summary>Gets or sets the Sep-STS blocks of the four stages, shallowest first (2, 2, 6, 2).</summary>
    public int[] StageDepths { get; set; } = { 2, 2, 6, 2 };

    /// <summary>Gets or sets the attention heads of the four stages, shallowest first (2, 4, 8, 16).</summary>
    public int[] StageHeads { get; set; } = { 2, 4, 8, 16 };

    /// <summary>Gets or sets the spatial attention window side (8); every second block shifts it by half.</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the MLP's hidden width as a multiple of the stage width (4).</summary>
    public double MlpRatio { get; set; } = 4.0;

    /// <summary>Gets or sets the synthesis features' width (nf_out, 64).</summary>
    public int SynthesisChannels { get; set; } = 64;

    /// <summary>Gets or sets the AdaCoF kernel side (ks, 5).</summary>
    public int KernelSize { get; set; } = 5;

    /// <summary>Gets or sets the AdaCoF kernel dilation (1).</summary>
    public int Dilation { get; set; } = 1;

    #endregion

    #region Training

    /// <summary>Gets or sets Adamax's initial learning rate (2e-4).</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets Adamax's first-moment decay (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adamax's infinity-norm decay (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets the batches at which the rate steps to the next of <see cref="LearningRateSteps"/>: epochs 40, 60, 75,
    /// 85 and 95 of the 16,153 batches of 4 in Vimeo-90K's 64,612 training septuplets (646,120, 969,180, 1,211,475,
    /// 1,373,005, 1,534,535).
    /// </summary>
    public int[] LearningRateMilestones { get; set; } = { 646120, 969180, 1211475, 1373005, 1534535 };

    /// <summary>Gets or sets the rates after each milestone (1e-4, 5e-5, 2.5e-5, 5e-6, 1e-6).</summary>
    public double[] LearningRateSteps { get; set; } = { 1e-4, 5e-5, 2.5e-5, 5e-6, 1e-6 };

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
