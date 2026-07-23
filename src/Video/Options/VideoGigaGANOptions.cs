using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VideoGigaGAN large-scale GAN for video SR.
/// </summary>
/// <remarks>
/// <para>
/// VideoGigaGAN (Xu et al., CVPR 2025) is the first large-scale GAN for video SR:
/// - GigaGAN backbone: upscaled StyleGAN architecture with 1B+ parameters, providing
///   exceptional detail generation capability beyond diffusion models in sharpness
/// - Feature propagation with anti-aliasing: temporal feature propagation using
///   anti-aliased flow warping to prevent temporal aliasing artifacts
/// - High-frequency shuttle: a dedicated pathway that extracts and preserves high-frequency
///   details (edges, textures) from the input through a parallel processing stream,
///   preventing the GAN from hallucinating false details
/// - Temporal discriminator: a 3D discriminator that evaluates temporal consistency
///   in addition to per-frame quality
/// - Supports up to 8x upscaling with rich perceptual details
/// </para>
/// <para>
/// <b>For Beginners:</b> VideoGigaGAN is like having a very talented artist with a
/// magnifying glass. While diffusion models (like StableVideoSR) gradually "develop"
/// a high-res image from noise, VideoGigaGAN directly "paints" detail in a single
/// forward pass -- much faster. A special "detail shuttle" ensures real details are
/// preserved while fake ones are avoided, and anti-aliasing prevents flickering.
/// </para>
/// </remarks>
public class VideoGigaGANOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public VideoGigaGANOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VideoGigaGANOptions(VideoGigaGANOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        ScaleFactor = other.ScaleFactor;
        NumStyleLayers = other.NumStyleLayers;
        PerceptualWeight = other.PerceptualWeight;
        GANWeight = other.GANWeight;
        HFShuttleWeight = other.HFShuttleWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels in the generator backbone.</summary>
    public int NumFeatures { get; set; } = 128;

    /// <summary>Gets or sets the number of residual blocks in the generator.</summary>
    public int NumResBlocks { get; set; } = 23;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    /// <remarks>VideoGigaGAN supports up to 8x. Default is 4x.</remarks>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of style mixing layers in the GigaGAN generator.</summary>
    public int NumStyleLayers { get; set; } = 14;

    /// <summary>Gets or sets the weight for the perceptual (LPIPS) loss component.</summary>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight for the GAN adversarial loss component.</summary>
    public double GANWeight { get; set; } = 0.1;

    /// <summary>Gets or sets the weight for the high-frequency shuttle loss.</summary>
    public double HFShuttleWeight { get; set; } = 0.5;

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
