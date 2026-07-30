using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for LiteDVDNet lightweight deep video denoising.
/// </summary>
/// <remarks>
/// <para>
/// LiteDVDNet (Ilchenko &amp; Stirenko, IJIGSP 2025) optimizes FastDVDnet for speed via four changes:
/// - Caching intermediate denoising results across overlapping frame windows (INFERENCE only - the paper
///   notes that mode "is suitable only for computations, not for network training")
/// - Reducing the InputCvBlock's intermediate channels by a factor of three (90 -&gt; 30)
/// - Simplifying each convolutional block to a single convolution ("LiteCvBlock")
/// - Halving the channel count for the smaller variant (2.48M -&gt; 0.64M parameters)
/// It keeps FastDVDnet's Conv -&gt; BatchNorm -&gt; ReLU ordering, its PixelShuffle upsampling, and its residual
/// connection between the central noisy input frame and the output.
/// NOTE: earlier documentation here described depthwise separable convolutions and 1x1 temporal fusion.
/// Neither appears anywhere in the paper.
/// </para>
/// <para>
/// <b>For Beginners:</b> LiteDVDNet is a fast, lightweight video denoiser. It first cleans
/// each frame individually, then combines information from nearby frames to improve quality.
/// It's designed to run efficiently on devices with limited computing power.
/// </para>
/// </remarks>
public class LiteDVDNetOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public LiteDVDNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LiteDVDNetOptions(LiteDVDNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        InputBlockIntermediateChannels = other.InputBlockIntermediateChannels;
        NumBlocks = other.NumBlocks;
        TemporalWindowSize = other.TemporalWindowSize;
        ExpansionFactor = other.ExpansionFactor;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        ResidualHeadInitScale = other.ResidualHeadInitScale;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Gets or sets the base feature width, which is the paper's variant number: 32 for LiteDVDNet-32 (the
    /// balanced variant, 3x faster than FastDVDnet for -0.18 dB PSNR) and 16 for LiteDVDNet-16 (5x faster,
    /// -0.61 dB). Halving it is the paper's fourth optimization and takes the model from 2.48M to 0.64M
    /// parameters. The previous default of 48 matched neither variant.
    /// </summary>
    public int NumFeatures { get; set; } = 32;

    /// <summary>
    /// Gets or sets the InputCvBlock's intermediate channel count. The paper reduces this by a factor of three,
    /// from FastDVDnet's 90 down to 30, which it identifies as one of the largest speedups available because the
    /// count directly determines how many convolutions run over the full-resolution input.
    /// </summary>
    public int InputBlockIntermediateChannels { get; set; } = 30;

    /// <summary>Gets or sets the number of denoising blocks per stage.</summary>
    public int NumBlocks { get; set; } = 4;

    /// <summary>Gets or sets the temporal window size (number of input frames).</summary>
    public int TemporalWindowSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets a legacy expansion factor. NOT part of the LiteDVDNet paper - it described a depthwise
    /// separable design this model never had. Retained so existing callers still compile; the paper architecture
    /// ignores it.
    /// </summary>
    public int ExpansionFactor { get; set; } = 2;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate. Default 1e-3 — the paper trains with "the ADAM algorithm with default
    /// hyperparameters", starting at 1e-3 and halving every 10 epochs after the first 10. This rate diverged
    /// against the previous architecture only because that stack had no batch normalization; with the paper's
    /// Conv -> BatchNorm -> ReLU ordering and the residual connection restored, it is the correct default.
    /// Fully user-overridable via this option or the constructor's optimizer parameter.
    /// </summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the scale applied to the residual-prediction head's initial weights. The network
    /// predicts the NOISE and the denoised frame is the input minus that estimate, so scaling the head
    /// down makes an untrained model start near the identity (predicts ~zero noise) — the standard prior
    /// for a residual denoiser ("having learned nothing, change nothing"). Default 0.01. Set to 1.0 to
    /// disable the damping and keep the head's raw initialization. Fully user-overridable.
    /// </summary>
    public double ResidualHeadInitScale { get; set; } = 0.01;

    #endregion
}
