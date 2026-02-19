using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the AVID (Audio-Visual Inpainting Diffusion) video inpainting model.
/// </summary>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "AVID: Any-Length Video Inpainting with Diffusion Model" (Zhang et al., CVPR 2024)</item>
/// </list></para>
/// <para>
/// AVID uses a diffusion-based approach for video inpainting, supporting arbitrary video lengths
/// through an autoregressive temporal pipeline with overlapping windows for consistency.
/// </para>
/// </remarks>
public class AVIDOptions : ModelOptions
{
    /// <summary>
    /// Model variant controlling capacity and speed trade-off.
    /// </summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>
    /// Number of base feature channels in the diffusion U-Net.
    /// </summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>
    /// Number of diffusion steps for the denoising process.
    /// </summary>
    public int NumDiffusionSteps { get; set; } = 50;

    /// <summary>
    /// Number of residual blocks per U-Net level.
    /// </summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>
    /// Number of attention heads in the temporal transformer layers.
    /// </summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>
    /// Size of overlapping temporal window for long video processing.
    /// </summary>
    public int TemporalOverlap { get; set; } = 4;

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
