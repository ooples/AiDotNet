using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for Real-ESRGAN extended to video with temporal consistency.
/// </summary>
/// <remarks>
/// <para>
/// Real-ESRGAN Video (Wang et al., 2022) extends the image-based Real-ESRGAN to video:
/// - RRDB backbone: Residual-in-Residual Dense Blocks (RRDBs) from ESRGAN provide the
///   per-frame feature extraction with strong representational capacity
/// - Second-order degradation model: simulates realistic degradations by applying
///   blur-resize-noise-JPEG twice in sequence, covering a wider range of real-world artifacts
/// - Temporal consistency module: flow-guided feature alignment between adjacent frames
///   with a temporal aggregation layer that fuses aligned features
/// - U-Net discriminator: a U-Net-based discriminator provides both global and local
///   adversarial feedback for high-quality perceptual results
/// </para>
/// <para>
/// <b>For Beginners:</b> Real-ESRGAN is one of the most popular practical upscaling tools.
/// The video version adds temporal awareness so that when you upscale a video, each frame
/// looks consistent with its neighbors (no flickering). It uses a realistic degradation
/// model during training, so it handles real-world issues like compression artifacts,
/// noise, and blur that lab models struggle with.
/// </para>
/// </remarks>
public class RealESRGANVideoOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RealESRGANVideoOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RealESRGANVideoOptions(RealESRGANVideoOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumRRDBBlocks = other.NumRRDBBlocks;
        ScaleFactor = other.ScaleFactor;
        DenseLayersPerBlock = other.DenseLayersPerBlock;
        ResidualScale = other.ResidualScale;
        PerceptualWeight = other.PerceptualWeight;
        GANWeight = other.GANWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of RRDB (Residual-in-Residual Dense Block) blocks.</summary>
    /// <remarks>The RealESRGAN-x4plus model uses 23 RRDBs.</remarks>
    public int NumRRDBBlocks { get; set; } = 23;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the number of dense layers per RRDB.</summary>
    public int DenseLayersPerBlock { get; set; } = 5;

    /// <summary>Gets or sets the residual scaling factor for RRDB stability.</summary>
    /// <remarks>Scales the residual connection in each RRDB to prevent training instability.</remarks>
    public double ResidualScale { get; set; } = 0.2;

    /// <summary>Gets or sets the weight for perceptual (LPIPS) loss.</summary>
    public double PerceptualWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the weight for GAN adversarial loss.</summary>
    public double GANWeight { get; set; } = 0.1;

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

    #endregion
}
