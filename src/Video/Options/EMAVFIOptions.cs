using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.EMAVFI{T}"/>, frame interpolation by
/// extracting motion and appearance via inter-frame attention (Zhang et al., CVPR 2023).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's main model (MCG-NJU/EMA-VFI, config.py with F = 32, W = 7 and depths
/// 2, 2, 2, 4, 4); <see cref="Small"/> returns EMA-VFI-small's. Training follows train.py and Trainer.py.
/// </para>
/// <para>
/// <b>For Beginners:</b> EMA-VFI compares small windows of the two frames with attention: what each window attends
/// to in the other frame describes its appearance, and where it attends describes its motion. Both feed a
/// coarse-to-fine motion estimator and a refinement network that produce the in-between frame.
/// </para>
/// </remarks>
public class EMAVFIOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public EMAVFIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EMAVFIOptions(EMAVFIOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        FeatureChannels = other.FeatureChannels;
        WindowSize = other.WindowSize;
        Depths = (int[])other.Depths.Clone();
        MlpRatio = other.MlpRatio;
        LearningRate = other.LearningRate;
        MinLearningRate = other.MinLearningRate;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        TotalIterations = other.TotalIterations;
        MergedWeight = other.MergedWeight;
        LaplacianLevels = other.LaplacianLevels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>EMA-VFI-small (config.py, F = 16, depths 2, 2, 2, 2, 2).</summary>
    public static EMAVFIOptions Small() => new()
    {
        FeatureChannels = 16,
        Depths = new[] { 2, 2, 2, 2, 2 },
    };

    #region Architecture

    /// <summary>
    /// Gets or sets F, the first stage's width; the five stages are F, 2F, 4F, 8F and 16F wide and the two attention
    /// stages have F/4 and F/2 heads (32).
    /// </summary>
    public int FeatureChannels { get; set; } = 32;

    /// <summary>Gets or sets the attention window's side W; odd blocks shift it by W / 2 (7).</summary>
    public int WindowSize { get; set; } = 7;

    /// <summary>
    /// Gets or sets the five stages' depths: convolutions in the first three, inter-frame attention blocks in the last
    /// two (2, 2, 2, 4, 4).
    /// </summary>
    public int[] Depths { get; set; } = { 2, 2, 2, 4, 4 };

    /// <summary>Gets or sets the attention blocks' MLP expansion (4).</summary>
    public double MlpRatio { get; set; } = 4.0;

    #endregion

    #region Training

    /// <summary>Gets or sets AdamW's peak learning rate (2e-4).</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the rate the cosine schedule ends at (2e-5).</summary>
    public double MinLearningRate { get; set; } = 2e-5;

    /// <summary>Gets or sets AdamW's weight decay (1e-4).</summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>Gets or sets the linear warm-up from 0 (2000 steps).</summary>
    public int WarmupSteps { get; set; } = 2000;

    /// <summary>
    /// Gets or sets the schedule's length: 300 epochs of Vimeo90K's 51,312 triplets at a total batch of 32 (four
    /// processes of 8).
    /// </summary>
    public int TotalIterations { get; set; } = 300 * (51312 / 32);

    /// <summary>Gets or sets the weight of each stage's warped blend in the loss (0.5).</summary>
    public double MergedWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the Laplacian loss's pyramid levels (5).</summary>
    public int LaplacianLevels { get; set; } = 5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
