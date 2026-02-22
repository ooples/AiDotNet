using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for UDVD unidirectional deep video denoising.
/// </summary>
/// <remarks>
/// <para>
/// UDVD (Sheth et al., CVPR 2021) performs blind video denoising without paired training data:
/// - Blind denoising: requires only noisy video for training (no clean ground truth),
///   using a self-supervised loss that exploits temporal redundancy
/// - Unidirectional: processes frames in a single forward pass using only past frames,
///   enabling real-time streaming operation
/// - Multi-frame fusion: combines features from multiple past frames with learned weights
///   that adapt to content and noise characteristics
/// - Noise-adaptive: handles varying and unknown noise levels including real camera noise
///   (not just synthetic Gaussian), making it practical for real-world footage
/// </para>
/// <para>
/// <b>For Beginners:</b> UDVD removes noise from video without needing clean reference
/// footage for training. It learns to denoise by recognizing that noise is random (different
/// each frame) while real content is consistent. This makes it especially useful for
/// real-world noisy video where you don't have a clean version to compare against.
/// </para>
/// </remarks>
public class UDVDOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public UDVDOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UDVDOptions(UDVDOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumLevels = other.NumLevels;
        NumResBlocks = other.NumResBlocks;
        TemporalBufferSize = other.TemporalBufferSize;
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

    /// <summary>Gets or sets the number of U-Net levels.</summary>
    public int NumLevels { get; set; } = 4;

    /// <summary>Gets or sets the number of residual blocks per level.</summary>
    public int NumResBlocks { get; set; } = 2;

    /// <summary>Gets or sets the temporal buffer size (past frames to use).</summary>
    public int TemporalBufferSize { get; set; } = 5;

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
