using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the MIA-VSR masked inter and intra-frame attention model.
/// </summary>
/// <remarks>
/// <para>
/// MIA-VSR (Zhou et al., CVPR 2024) uses masked attention for efficient video SR:
/// - Masked inter-frame attention: temporal attention across frames with sparse masking,
///   attending only to the most relevant spatial locations in neighboring frames
/// - Masked intra-frame attention: spatial attention within each frame with local window
///   masking for computational efficiency
/// - Progressive masking: the masking ratio decreases through layers, from coarse to fine
/// - Built on BasicVSR++ backbone with attention replacing deformable convolution
/// </para>
/// <para>
/// <b>For Beginners:</b> MIA-VSR makes video super-resolution faster by being selective
/// about what it pays attention to. Instead of looking at every pixel in every frame
/// (which is slow), it uses "masks" to focus only on the most important parts. It looks
/// between frames (inter) to track moving objects and within frames (intra) to enhance
/// spatial details.
/// </para>
/// </remarks>
public class MIAVSROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public MIAVSROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MIAVSROptions(MIAVSROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumResBlocks = other.NumResBlocks;
        ScaleFactor = other.ScaleFactor;
        WindowSize = other.WindowSize;
        NumHeads = other.NumHeads;
        InterMaskRatio = other.InterMaskRatio;
        IntraMaskRatio = other.IntraMaskRatio;
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

    /// <summary>Gets or sets the number of residual blocks in each propagation branch.</summary>
    public int NumResBlocks { get; set; } = 30;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the window size for masked intra-frame attention.</summary>
    /// <remarks>Spatial attention is computed within non-overlapping windows of this size.</remarks>
    public int WindowSize { get; set; } = 8;

    /// <summary>Gets or sets the number of attention heads for inter/intra-frame attention.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the masking ratio for inter-frame attention (0.0-1.0).</summary>
    /// <remarks>Higher values mask more locations, reducing computation but potentially missing details.</remarks>
    public double InterMaskRatio { get; set; } = 0.5;

    /// <summary>Gets or sets the masking ratio for intra-frame attention (0.0-1.0).</summary>
    public double IntraMaskRatio { get; set; } = 0.25;

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
