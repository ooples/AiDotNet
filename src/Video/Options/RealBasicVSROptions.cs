using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RealBasicVSR real-world video super-resolution model.
/// </summary>
/// <remarks>
/// <para>
/// RealBasicVSR (Chan et al., CVPR 2022) addresses real-world video SR through:
/// - Stochastic degradation scheme: random combination of blur, noise, resize, and compression
///   during training to handle unknown real-world degradations
/// - Pre-cleaning module: a lightweight network that removes noise/artifacts before the
///   BasicVSR backbone, preventing degradation from propagating across frames
/// - Dynamic refinement: the cleaning module strength adapts to the degradation level
/// </para>
/// <para>
/// <b>For Beginners:</b> Real videos have unpredictable quality issues (noise, blur, compression).
/// RealBasicVSR adds a "cleaning" step before upscaling that removes these artifacts first,
/// then uses the BasicVSR backbone for high-quality super-resolution.
/// </para>
/// </remarks>
public class RealBasicVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RealBasicVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RealBasicVSROptions(RealBasicVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        CleaningModuleBlocks = other.CleaningModuleBlocks;
        ScaleFactor = other.ScaleFactor;
        NumFrames = other.NumFrames;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        WarmupSteps = other.WarmupSteps;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of residual blocks in the BasicVSR backbone.</summary>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the number of residual blocks in the pre-cleaning module.</summary>
    /// <remarks>The cleaning module processes each frame independently before propagation.</remarks>
    public int CleaningModuleBlocks { get; set; } = 20;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of input frames.</summary>
    public int NumFrames { get; set; } = 15;

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

    /// <summary>Gets or sets the warmup steps.</summary>
    public int WarmupSteps { get; set; } = 5000;

    #endregion
}
