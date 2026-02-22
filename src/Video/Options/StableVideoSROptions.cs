using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the StableVideoSR temporal-conditioned diffusion video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// StableVideoSR (2024) adapts the Stable Diffusion architecture for video SR:
/// - Temporal conditioning modules: inserted between spatial attention layers in the U-Net,
///   these cross-attend to features from adjacent frames for temporal coherence
/// - ControlNet adapter: a frozen copy of the encoder provides fine-grained spatial control
///   from the low-resolution input while the main U-Net generates high-resolution output
/// - Classifier-free guidance: balances restoration fidelity vs generative quality
/// </para>
/// <para>
/// <b>For Beginners:</b> StableVideoSR takes the popular Stable Diffusion image AI and
/// extends it to handle video. It adds special "temporal" modules that look at neighboring
/// frames to ensure the output video is smooth and flicker-free, not just a sequence
/// of independently upscaled images.
/// </para>
/// </remarks>
public class StableVideoSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public StableVideoSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public StableVideoSROptions(StableVideoSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumDenoisingSteps = other.NumDenoisingSteps;
        NumTemporalLayers = other.NumTemporalLayers;
        ScaleFactor = other.ScaleFactor;
        GuidanceScale = other.GuidanceScale;
        ControlNetScale = other.ControlNetScale;
        LatentDim = other.LatentDim;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of UNet feature channels.</summary>
    public int NumFeatures { get; set; } = 320;

    /// <summary>Gets or sets the number of denoising steps.</summary>
    public int NumDenoisingSteps { get; set; } = 20;

    /// <summary>Gets or sets the number of temporal attention layers.</summary>
    public int NumTemporalLayers { get; set; } = 4;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the classifier-free guidance scale.</summary>
    public double GuidanceScale { get; set; } = 7.5;

    /// <summary>Gets or sets the ControlNet conditioning scale.</summary>
    public double ControlNetScale { get; set; } = 1.0;

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
