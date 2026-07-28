using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for BSVD bidirectional streaming video denoising.
/// </summary>
/// <remarks>
/// <para>
/// BSVD (Qi et al., ACM MM 2022) enables real-time video denoising through bidirectional
/// streaming with efficient buffer management:
/// - Bidirectional streaming: processes video in both forward and backward passes with shared
///   buffers, so each frame benefits from both past and future context
/// - Streaming buffers: maintains compact latent buffers instead of storing full frames,
///   enabling constant-memory processing regardless of video length
/// - Real-time capability: designed for 30+ fps denoising on consumer GPUs through
///   efficient buffer reuse and single-pass-per-direction processing
/// - Noise-adaptive: handles varying noise levels without requiring noise-level input,
///   making it suitable for real-world video with spatially varying noise
/// </para>
/// <para>
/// <b>For Beginners:</b> BSVD cleans up noisy video in real-time by looking at both past and
/// future frames. Unlike methods that need all frames at once, it processes them in a stream
/// using small memory buffers, making it practical for live video and long recordings.
/// </para>
/// </remarks>
public class BSVDOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public BSVDOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BSVDOptions(BSVDOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumRecurrentBlocks = other.NumRecurrentBlocks;
        BufferDim = other.BufferDim;
        NumLevels = other.NumLevels;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of recurrent blocks per direction.</summary>
    public int NumRecurrentBlocks { get; set; } = 4;

    /// <summary>Gets or sets the hidden state dimension for streaming buffers.</summary>
    public int BufferDim { get; set; } = 64;

    /// <summary>Gets or sets the number of U-Net encoder/decoder levels.</summary>
    public int NumLevels { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of sequential U-Nets in the denoising backbone.
    /// </summary>
    /// <value>Defaults to 2 — the paper's W-Net.</value>
    /// <remarks>
    /// <para>Qi et al. 2022 (arXiv:2207.06937) build the backbone from "two light-weight U-Nets"
    /// forming a W-Net, and their ablation measures +0.40 dB for the second stage over a single
    /// U-Net. The shared video-denoising factory previously used here emitted only one
    /// encoder/decoder stack.</para>
    /// <para><b>For Beginners:</b> The video is cleaned twice in a row — the second pass refines
    /// what the first left behind. Setting this to 1 gives the cheaper single-pass model.</para>
    /// </remarks>
    public int NumUNetStages { get; set; } = 2;

    /// <summary>
    /// Gets or sets the shifted-channel ratio <c>r</c> used by temporal fusion.
    /// </summary>
    /// <value>Defaults to 8, the paper's value.</value>
    /// <remarks>
    /// Per the paper, <c>floor(C_f / r)</c> channels shift per direction, so a larger <c>r</c>
    /// mixes fewer channels across time.
    /// </remarks>
    public int ShiftedChannelRatio { get; set; } = 8;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    /// <value>Defaults to 1e-3, the paper's initial Adam learning rate.</value>
    /// <remarks>
    /// Qi et al. 2022 (arXiv:2207.06937) train with Adam at an initial 1e-3, decayed by 0.7 every
    /// 50,000 iterations over 700,000 iterations. This previously defaulted to 1e-4, an order of
    /// magnitude below the published value.
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
