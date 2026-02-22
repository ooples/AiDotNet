using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the SeedVR diffusion transformer video restoration model.
/// </summary>
/// <remarks>
/// <para>
/// SeedVR (Wang et al., 2025) uses a diffusion transformer (DiT) architecture for generic
/// video restoration (SR, denoising, deblurring, JPEG artifact removal):
/// - Shifted window attention: efficient spatio-temporal self-attention with window shifting
///   for cross-window interaction (similar to Swin Transformer but in 3D)
/// - Causal temporal attention: maintains temporal consistency without future frame access
/// - Progressive upsampling: multi-stage 2x upsampling from latent to pixel space
/// - Text-to-video diffusion priors: initialized from a pretrained T2V model
/// </para>
/// <para>
/// <b>For Beginners:</b> SeedVR uses a powerful video generation AI model and adapts it
/// to fix degraded videos. It works like a smart upscaler that can handle many types
/// of damage (noise, blur, compression) because it learned from millions of clean videos
/// what natural video should look like.
/// </para>
/// </remarks>
public class SeedVROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public SeedVROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SeedVROptions(SeedVROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDiTBlocks = other.NumDiTBlocks;
        PatchSize = other.PatchSize;
        WindowSize = other.WindowSize;
        NumHeads = other.NumHeads;
        NumDenoisingSteps = other.NumDenoisingSteps;
        ScaleFactor = other.ScaleFactor;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels in the DiT blocks.</summary>
    public int NumFeatures { get; set; } = 192;

    /// <summary>Gets or sets the number of DiT transformer blocks.</summary>
    public int NumDiTBlocks { get; set; } = 24;

    /// <summary>Gets or sets the patch size for tokenizing video frames.</summary>
    public int PatchSize { get; set; } = 2;

    /// <summary>Gets or sets the local window size for shifted window attention.</summary>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the number of denoising steps.</summary>
    public int NumDenoisingSteps { get; set; } = 50;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>Gets or sets the warmup steps.</summary>
    public int WarmupSteps { get; set; } = 5000;

    #endregion
}
