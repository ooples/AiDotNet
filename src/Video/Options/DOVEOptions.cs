using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the DOVE video diffusion prior restoration model.
/// </summary>
/// <remarks>
/// <para>
/// DOVE (Chen et al., 2025) harnesses large-scale video diffusion models as priors:
/// - Degradation estimation stage: analyzes the input to determine degradation type/level
/// - Guided generation stage: conditions a pretrained video diffusion model (SVD backbone)
///   on the degradation estimate for restoration-aware denoising
/// - Temporal consistency: video diffusion priors inherently maintain frame coherence
/// </para>
/// <para>
/// <b>For Beginners:</b> DOVE uses a large video generation AI model (similar to those
/// that create videos from text) but repurposes it for restoration. Instead of generating
/// new content, it generates clean versions of degraded videos by leveraging the model's
/// knowledge of what natural video looks like.
/// </para>
/// </remarks>
public class DOVEOptions : NeuralNetworkOptions
{
    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of UNet feature channels.</summary>
    public int NumFeatures { get; set; } = 320;

    /// <summary>Gets or sets the number of denoising steps in the diffusion process.</summary>
    public int NumDenoisingSteps { get; set; } = 20;

    /// <summary>Gets or sets the number of residual blocks per UNet level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the number of attention heads in the UNet.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the classifier-free guidance scale.</summary>
    /// <remarks>Higher values increase fidelity to the degradation estimate but may reduce diversity.</remarks>
    public double GuidanceScale { get; set; } = 7.5;

    /// <summary>Gets or sets the latent space dimension.</summary>
    public int LatentDim { get; set; } = 4;

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
