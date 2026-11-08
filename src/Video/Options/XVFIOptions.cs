using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.XVFI{T}"/>, extreme video frame interpolation
/// (Sim, Oh and Kim, ICCV 2021).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the release's X4K1000FPS model (JihyongOh/XVFI: XVFInet.py, main.py, utils.py): 64 feature channels,
/// a 4x spatial reduction, three scale levels below the input in training and five at test; <see cref="Vimeo"/> returns
/// XVFI-v, the Vimeo90K model (2x reduction, one level). Training follows main.py: Adam at 1e-4, divided by 4 at epochs
/// 100, 150 and 180, on the multi-scale L1 reconstruction loss plus half the edge-aware flow smoothness.
/// </para>
/// <para>
/// <b>For Beginners:</b> XVFI handles very large motion in 4K video by estimating motion on a pyramid of ever smaller
/// copies of the frames, starting at the coarsest, and refining it one level at a time. The same small network is reused
/// at every level, so you can add levels at test time for bigger motion.
/// </para>
/// </remarks>
public class XVFIOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public XVFIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public XVFIOptions(XVFIOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        NumFeatures = other.NumFeatures;
        ScaleFactor = other.ScaleFactor;
        TrainingDepth = other.TrainingDepth;
        TestDepth = other.TestDepth;
        LearningRate = other.LearningRate;
        LearningRateMilestones = (int[])other.LearningRateMilestones.Clone();
        LearningRateDecay = other.LearningRateDecay;
        ReconstructionWeight = other.ReconstructionWeight;
        SmoothnessWeight = other.SmoothnessWeight;
        EdgeConstant = other.EdgeConstant;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>XVFI-v, the release's Vimeo90K configuration: a 2x reduction and one scale level in training and test.</summary>
    public static XVFIOptions Vimeo() => new() { ScaleFactor = 2, TrainingDepth = 1, TestDepth = 1 };

    #region Architecture

    /// <summary>Gets or sets the feature channels (nf, 64).</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the feature extractor's spatial reduction, a power of two (module_scale_factor, 4).</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the lowest scale level in training (S_trn, 3).</summary>
    public int TrainingDepth { get; set; } = 3;

    /// <summary>Gets or sets the lowest scale level at test (S_tst, 5).</summary>
    public int TestDepth { get; set; } = 5;

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's initial learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the epochs at which the rate is multiplied by <see cref="LearningRateDecay"/> (100, 150, 180).</summary>
    public int[] LearningRateMilestones { get; set; } = { 100, 150, 180 };

    /// <summary>Gets or sets the factor applied at each milestone (0.25).</summary>
    public double LearningRateDecay { get; set; } = 0.25;

    /// <summary>Gets or sets the weight of the multi-scale reconstruction loss (rec_lambda, 1).</summary>
    public double ReconstructionWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the edge-aware flow smoothness loss (0.5).</summary>
    public double SmoothnessWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the smoothness loss's edge constant, squared in its exponent (150).</summary>
    public double EdgeConstant { get; set; } = 150.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
