using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for SwinVFI Swin Transformer-based video frame interpolation.
/// </summary>
/// <remarks>
/// <para>
/// SwinVFI (2022) applies Swin Transformer architecture to frame interpolation:
/// - Swin Transformer encoder: uses shifted-window self-attention to encode input frame pairs
///   with linear complexity (O(N) vs O(N^2) for full attention), enabling processing of
///   high-resolution frames without excessive memory
/// - Cross-frame window attention: extends Swin's shifted-window mechanism to cross-attend
///   between features from the two input frames, capturing inter-frame correspondences within
///   each local window and globally through window shifting
/// - Hierarchical feature pyramid: multi-scale feature extraction with Swin blocks at each
///   level, capturing both fine-grained texture details and large-scale motion context
/// - Flow-free synthesis: directly synthesizes the intermediate frame from cross-attended
///   features without explicit optical flow estimation, avoiding flow-related artifacts
/// </para>
/// <para>
/// <b>For Beginners:</b> SwinVFI uses the Swin Transformer (a powerful attention-based
/// architecture) to look at both input frames simultaneously and figure out what goes between
/// them, without needing to estimate motion explicitly. The "shifted window" approach makes it
/// efficient enough to handle full-resolution video frames.
/// </para>
/// </remarks>
public class SwinVFIOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public SwinVFIOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SwinVFIOptions(SwinVFIOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumSwinBlocks = other.NumSwinBlocks;
        NumHeads = other.NumHeads;
        WindowSize = other.WindowSize;
        NumStages = other.NumStages;
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

    /// <summary>Gets or sets the number of Swin Transformer blocks per stage.</summary>
    public int NumSwinBlocks { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the window size for shifted-window attention.</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the number of hierarchical stages.</summary>
    public int NumStages { get; set; } = 4;

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
