using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RealisVSR detail-enhanced diffusion 4K video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// RealisVSR (2025) achieves coherent 4K real-world video super-resolution through:
/// - Wan 2.1 video diffusion backbone with detail-enhancement ControlNet adapter
/// - Motion-aware temporal conditioning for consistent inter-frame motion
/// - Detail-enhancement module that preserves fine textures during the diffusion process
/// - Designed specifically for upscaling real-world degraded video to 4K resolution
/// </para>
/// <para>
/// <b>For Beginners:</b> RealisVSR uses an AI video generation model (Wan 2.1) to upscale
/// real-world video to 4K. It adds a special "detail enhancement" module that makes sure
/// fine details like text and textures stay sharp during the upscaling process, while the
/// video generation backbone ensures smooth, natural-looking motion.
/// </para>
/// </remarks>
public class RealisVSROptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of UNet feature channels.</summary>
    public int NumFeatures { get; set; } = 320;

    /// <summary>Gets or sets the number of denoising steps.</summary>
    public int NumDenoisingSteps { get; set; } = 25;

    /// <summary>Gets or sets the number of residual blocks per UNet level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the ControlNet conditioning scale for detail enhancement.</summary>
    public double ControlNetScale { get; set; } = 1.0;

    /// <summary>Gets or sets the classifier-free guidance scale.</summary>
    public double GuidanceScale { get; set; } = 7.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 5e-5;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the warmup steps.</summary>
    public int WarmupSteps { get; set; } = 1000;

    #endregion
}
