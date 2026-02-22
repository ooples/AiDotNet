using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FuseFormer video inpainting model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting" (Liu et al., ICCV 2021)</item>
/// </list></para>
/// <para><b>For Beginners:</b> FuseFormer options configure the transformer video inpainting model.</para>
/// <para>
/// FuseFormer uses a transformer architecture with soft split and composition operations
/// to fuse fine-grained spatial-temporal information at multiple scales for high-quality
/// video inpainting with better detail preservation.
/// </para>
/// </remarks>
public class FuseFormerOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public FuseFormerOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FuseFormerOptions(FuseFormerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumTransformerLayers = other.NumTransformerLayers;
        NumHeads = other.NumHeads;
        PatchSize = other.PatchSize;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>
    /// Model variant controlling capacity and speed trade-off.
    /// </summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Number of base feature channels.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Number of transformer layers for temporal fusion.
    /// </summary>
    public int NumTransformerLayers { get; set; } = 8;

    /// <summary>
    /// Number of attention heads in the multi-head attention modules.
    /// </summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Patch size for the soft split operation that divides features into overlapping patches.
    /// </summary>
    public int PatchSize { get; set; } = 3;

    /// <summary>
    /// Learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Path to the ONNX model file for inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options for inference mode.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
