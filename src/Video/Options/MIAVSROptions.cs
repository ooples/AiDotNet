using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for MIA-VSR, the masked inter- and intra-frame attention video
/// super-resolution transformer (Zhou et al., CVPR 2024).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the paper's: 120 channels, 8x8 windows, 6 heads, an FFN ratio of 2, four
/// propagation branches (backward, forward, backward, forward) of six inter-and-intra-frame attention
/// blocks each, and a mask-sparsity weight of 5e-4 with a Gumbel-softmax temperature of 2/3.
/// </para>
/// <para>
/// <b>For Beginners:</b> MIA-VSR upscales a video one frame at a time while looking back at the frames
/// it has already enhanced. Each attention block learns a mask that says which positions changed since
/// the previous frame; positions that did not change reuse last frame's result instead of being
/// recomputed, which is where the method saves its compute.
/// </para>
/// </remarks>
public class MIAVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public MIAVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MIAVSROptions(MIAVSROptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        ScaleFactor = other.ScaleFactor;
        WindowSize = other.WindowSize;
        NumHeads = other.NumHeads;
        FeedForwardRatio = other.FeedForwardRatio;
        NumPropagationBranches = other.NumPropagationBranches;
        BlocksPerBranch = other.BlocksPerBranch;
        MaskLossWeight = other.MaskLossWeight;
        GumbelTemperature = other.GumbelTemperature;
        ReconstructionChannels = other.ReconstructionChannels;
#pragma warning disable CS0618 // Retained only so existing configurations keep round-tripping.
        NumResBlocks = other.NumResBlocks;
        InterMaskRatio = other.InterMaskRatio;
        IntraMaskRatio = other.IntraMaskRatio;
#pragma warning restore CS0618
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the feature width C of every attention block.</summary>
    /// <value>Default is 120, the paper's. Must be divisible by <see cref="NumHeads"/>.</value>
    public int NumFeatures { get; set; } = 120;

    /// <summary>Gets or sets the spatial upscaling factor (a power of two).</summary>
    /// <value>Default is 4, the factor every reported result uses.</value>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the side of the square attention window.</summary>
    /// <value>Default is 8 (8x8 windows), the paper's. Frames are zero-padded to a multiple of it.</value>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the number of attention heads.</summary>
    /// <value>Default is 6, the paper's (head width 120 / 6 = 20).</value>
    public int NumHeads { get; set; } = 6;

    /// <summary>Gets or sets the FFN hidden width as a multiple of <see cref="NumFeatures"/>.</summary>
    /// <value>Default is 2: the paper's unmasked linear cost of 12·HWC² leaves 2r = 4 for the FFN.</value>
    public int FeedForwardRatio { get; set; } = 2;

    /// <summary>Gets or sets the number of feature propagation modules (branches).</summary>
    /// <value>Default is 4, BasicVSR++'s second-order grid: backward, forward, backward, forward.</value>
    public int NumPropagationBranches { get; set; } = 4;

    /// <summary>Gets or sets the number of inter-and-intra-frame attention blocks per branch.</summary>
    /// <value>Default is 6, the paper's skip-connection interval [6, 6, 6, 6].</value>
    public int BlocksPerBranch { get; set; } = 6;

    /// <summary>Gets or sets λ, the weight of the mask-sparsity loss added to the training objective.</summary>
    /// <value>Default is 5e-4, the setting the paper highlights (Table 1).</value>
    public double MaskLossWeight { get; set; } = 5e-4;

    /// <summary>Gets or sets τ, the temperature of the training-time Gumbel-softmax mask.</summary>
    /// <value>Default is 2/3, the paper's.</value>
    public double GumbelTemperature { get; set; } = 2.0 / 3.0;


    /// <summary>Gets or sets the channel width of the pixel-shuffle reconstruction head.</summary>
    /// <value>Default is 64, BasicVSR++'s upsampler width, which MIA-VSR's reconstruction follows.</value>
    public int ReconstructionChannels { get; set; } = 64;

    /// <summary>Not used: MIA-VSR has no residual convolution blocks.</summary>
    [Obsolete("MIA-VSR is built from inter-and-intra-frame attention blocks; use BlocksPerBranch and NumPropagationBranches. This value is ignored.")]
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Not used: MIA-VSR learns its masks; there is no fixed masking ratio.</summary>
    [Obsolete("MIA-VSR's masks are predicted per position and trained with MaskLossWeight; this value is ignored.")]
    public double InterMaskRatio { get; set; } = 0.5;

    /// <summary>Not used: MIA-VSR learns its masks; there is no fixed masking ratio.</summary>
    [Obsolete("MIA-VSR's masks are predicted per position and trained with MaskLossWeight; this value is ignored.")]
    public double IntraMaskRatio { get; set; } = 0.25;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    /// <remarks>MIA-VSR applies no dropout; the default 0 keeps it off.</remarks>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}