using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for RealBasicVSR, real-world video super-resolution with an image-cleaning module.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Chan et al., CVPR 2022, and the authors' release (open-mmlab/mmagic, RealBasicVSRNet and
/// configs/real_basicvsr/realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py, the PSNR-oriented first stage):
/// 64 channels, 20 propagation and 20 cleaning blocks, 15-frame clips, L1 output and cleaning losses against the
/// unsharp-masked target, Adam (0.9, 0.99) at 1e-4 without a schedule, SPyNet frozen, and a 0.999 moving average of the
/// weights for inference.
/// </para>
/// <para>
/// <b>For Beginners:</b> RealBasicVSR first "cleans" each frame (removing noise, blur and compression artifacts),
/// then upscales the cleaned frames with BasicVSR, so the artifacts are not spread from frame to frame.
/// </para>
/// </remarks>
public class RealBasicVSROptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RealBasicVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RealBasicVSROptions(RealBasicVSROptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        MidChannels = other.MidChannels;
        NumPropagationBlocks = other.NumPropagationBlocks;
        NumCleaningBlocks = other.NumCleaningBlocks;
        NumFrames = other.NumFrames;
        InferenceCleaningThreshold = other.InferenceCleaningThreshold;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        PixelLossWeight = other.PixelLossWeight;
        CleaningLossWeight = other.CleaningLossWeight;
        SharpenTarget = other.SharpenTarget;
        UnsharpWeight = other.UnsharpWeight;
        UnsharpKernelSize = other.UnsharpKernelSize;
        UnsharpThreshold = other.UnsharpThreshold;
        EmaMomentum = other.EmaMomentum;
    }

    #region Architecture

    /// <summary>Gets or sets the feature channels (64).</summary>
    public int MidChannels { get; set; } = 64;

    /// <summary>Gets or sets the residual blocks per propagation branch (20).</summary>
    public int NumPropagationBlocks { get; set; } = 20;

    /// <summary>Gets or sets the residual blocks of the cleaning module (20).</summary>
    public int NumCleaningBlocks { get; set; } = 20;

    /// <summary>Gets or sets the frames of a training clip (15).</summary>
    public int NumFrames { get; set; } = 15;

    /// <summary>
    /// Gets or sets the mean absolute cleaning residue, in pixel levels 0-255, below which inference stops cleaning
    /// (1.5, the release's test setting; training uses 255, i.e. one pass).
    /// </summary>
    public double InferenceCleaningThreshold { get; set; } = 1.5;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight of the L1 loss on the output (1.0).</summary>
    public double PixelLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight of the L1 loss of the cleaned frames against the 4x area-downsampled target (1.0).</summary>
    public double CleaningLossWeight { get; set; } = 1.0;

    /// <summary>Gets or sets whether the losses score the unsharp-masked target (true, is_use_sharpened_gt_in_pixel).</summary>
    public bool SharpenTarget { get; set; } = true;

    /// <summary>Gets or sets the unsharp-masking weight (0.5).</summary>
    public double UnsharpWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the unsharp-masking Gaussian size (51).</summary>
    public int UnsharpKernelSize { get; set; } = 51;

    /// <summary>Gets or sets the unsharp-masking residue threshold in pixel levels (10).</summary>
    public double UnsharpThreshold { get; set; } = 10;

    /// <summary>
    /// Gets or sets the momentum of the weights' moving average used for inference (0.001); null disables it and
    /// inference uses the trained weights.
    /// </summary>
    public double? EmaMomentum { get; set; } = 0.001;

    #endregion
}
