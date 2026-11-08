using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.InterpAnyClearer{T}"/>, distance indexing and
/// iterative reference-based estimation for clearer anytime frame interpolation (Zhong et al., ECCV 2024).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's DR-RIFE (zzh-tech/InterpAny-Clearer, models/DI-RIFE: IFNet_sdi_recur.py, refine.py,
/// RIFE_sdi_recur.py, train_sdi_m_mask_recur.py): RIFE's IFBlocks of widths 240, 150 and 90 at scales 4, 2 and 1, a
/// width-90 privileged teacher, the width-16 Contextnet and U-Net, each block reading the five-channel index map (target
/// distance, reference distance, reference frame), and two reference iterations at inference. Training follows the
/// release: AdamW with a weight decay of 1e-3, the rate rising linearly to 3e-4 over 2,000 steps and falling by a cosine
/// to 3e-6 over 300 epochs, on the Laplacian loss of the frame and of the teacher plus 0.002 times the distillation loss.
/// </para>
/// <para>
/// <b>For Beginners:</b> Ordinary interpolation is told only "halfway in time", which is ambiguous when objects speed up
/// or slow down, so the result blurs. This model is instead told, for every pixel, how far along its path the object
/// should be (the distance index). It reaches that position in a few small steps, each step starting from the frame the
/// previous step produced, which keeps the result sharp. A map that is the same everywhere behaves like a time.
/// </para>
/// </remarks>
public class InterpAnyClearerOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public InterpAnyClearerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public InterpAnyClearerOptions(InterpAnyClearerOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        BlockWidths = (int[])other.BlockWidths.Clone();
        TeacherWidth = other.TeacherWidth;
        ContextChannels = other.ContextChannels;
        UseDistillation = other.UseDistillation;
        Iterations = other.Iterations;
        LearningRate = other.LearningRate;
        MinLearningRate = other.MinLearningRate;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        TotalSteps = other.TotalSteps;
        DistillationWeight = other.DistillationWeight;
        LaplacianLevels = other.LaplacianLevels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>Gets or sets the three student IFBlocks' widths, coarsest first (240, 150, 90); each is even.</summary>
    public int[] BlockWidths { get; set; } = { 240, 150, 90 };

    /// <summary>Gets or sets the privileged teacher IFBlock's width (90), even.</summary>
    public int TeacherWidth { get; set; } = 90;

    /// <summary>Gets or sets the Contextnet's and U-Net's base width c (16).</summary>
    public int ContextChannels { get; set; } = 16;

    /// <summary>Gets or sets whether the teacher block exists and distills the students in training (true).</summary>
    public bool UseDistillation { get; set; } = true;

    /// <summary>Gets or sets the reference iterations from frame 0 to the target index map at inference (iters, 2).</summary>
    public int Iterations { get; set; } = 2;

    #endregion

    #region Training

    /// <summary>Gets or sets AdamW's peak learning rate, reached after the warmup (3e-4).</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gets or sets the rate the cosine decays to (3e-6).</summary>
    public double MinLearningRate { get; set; } = 3e-6;

    /// <summary>Gets or sets AdamW's decoupled weight decay (1e-3).</summary>
    public double WeightDecay { get; set; } = 1e-3;

    /// <summary>Gets or sets the steps over which the rate rises linearly from zero (2,000).</summary>
    public int WarmupSteps { get; set; } = 2000;

    /// <summary>
    /// Gets or sets the step at which the cosine reaches <see cref="MinLearningRate"/>: 300 epochs of the septuplet
    /// training set at the release's 1,009 steps per epoch (302,700).
    /// </summary>
    public int TotalSteps { get; set; } = 302700;

    /// <summary>Gets or sets the distillation loss's weight (0.002, the continuous-index setting).</summary>
    public double DistillationWeight { get; set; } = 0.002;

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
