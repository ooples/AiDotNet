using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the STTN (Spatial-Temporal Transformer Network) video inpainting model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Learning Joint Spatial-Temporal Transformations for Video Inpainting" (Zeng et al., ECCV 2020)</item>
/// </list></para>
/// <para><b>For Beginners:</b> STTN options configure the spatial-temporal transformer inpainting network.</para>
/// <para>
/// STTN uses multi-scale spatial-temporal transformers to simultaneously search for and
/// attend to relevant patches across space and time, filling masked regions with content
/// from visible areas in the same and other frames.
/// </para>
/// </remarks>
public class STTNOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public STTNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public STTNOptions(STTNOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumTransformerLayers = other.NumTransformerLayers;
        NumHeads = other.NumHeads;
        NumScales = other.NumScales;
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
    /// Number of spatial-temporal transformer layers.
    /// </summary>
    public int NumTransformerLayers { get; set; } = 8;

    /// <summary>
    /// Number of attention heads for multi-head spatial-temporal attention.
    /// </summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Number of scales for multi-scale patch matching.
    /// </summary>
    public int NumScales { get; set; } = 3;

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
