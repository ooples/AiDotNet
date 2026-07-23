using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for TLBVFI token-level bidirectional video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// TLBVFI (2024) operates at the token level for bidirectional frame interpolation:
/// - Token-level processing: divides input frames into non-overlapping tokens (patches) and
///   performs all operations (flow estimation, feature extraction, synthesis) at the token level,
///   enabling efficient processing of high-resolution frames with transformer architectures
/// - Bidirectional token matching: for each target-time token, finds corresponding tokens in
///   both input frames using learned token-level cross-attention, naturally handling both forward
///   and backward motion in a single pass
/// - Token-level flow: estimates optical flow at the token (patch) level rather than the pixel
///   level, which is more robust to noise and local ambiguities while being computationally
///   cheaper, with sub-token refinement applied after initial matching
/// - Adaptive token merging: dynamically merges tokens in low-motion regions to reduce
///   computation, while keeping fine-grained tokens in high-motion areas for accuracy
/// </para>
/// <para>
/// <b>For Beginners:</b> TLBVFI breaks each frame into small patches (tokens) and works with
/// these patches instead of individual pixels. This is faster and more robust, like reading
/// words instead of individual letters. It finds matching patches in both frames and uses them
/// to build the intermediate frame.
/// </para>
/// </remarks>
public class TLBVFIOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public TLBVFIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TLBVFIOptions(TLBVFIOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        TokenSize = other.TokenSize;
        NumMatchingBlocks = other.NumMatchingBlocks;
        NumHeads = other.NumHeads;
        NumSynthesisBlocks = other.NumSynthesisBlocks;
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

    /// <summary>Gets or sets the token (patch) size in pixels.</summary>
    public int TokenSize { get; set; } = 8;

    /// <summary>Gets or sets the number of bidirectional matching blocks.</summary>
    public int NumMatchingBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of attention heads for token matching.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the number of synthesis refinement blocks.</summary>
    public int NumSynthesisBlocks { get; set; } = 3;

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
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
