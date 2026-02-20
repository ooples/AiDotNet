using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the GaVS (Gaze-aware Video Stabilization) model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Gaze-aware Video Stabilization" (2023)</item>
/// </list></para>
/// <para>
/// GaVS incorporates gaze prediction to stabilize video while preserving the viewer's
/// region of interest, weighting stabilization strength based on visual saliency.
/// </para>
/// </remarks>
public class GaVSOptions : ModelOptions
{
    /// <summary>
    /// Model variant controlling capacity and speed trade-off.
    /// </summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Number of base feature channels.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Number of gaze prediction heads for multi-scale saliency estimation.
    /// </summary>
    public int NumGazeHeads { get; set; } = 4;

    /// <summary>
    /// Hidden dimension for the gaze prediction branch.
    /// </summary>
    public int GazeHiddenDim { get; set; } = 128;

    /// <summary>
    /// Smoothing window size for trajectory filtering.
    /// </summary>
    public int SmoothingWindow { get; set; } = 30;

    /// <summary>
    /// Learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Path to the ONNX model file for inference mode.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options for inference mode.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
