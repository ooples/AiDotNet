using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for UDVD, unsupervised deep video denoising with a blind-spot network (Sheth et al.,
/// ICCV 2021).
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the paper and the authors' release (github.com/sreyas-mohan/udvd, models/udvd.py,
/// "blind-video-net-4"): five frames, two blind-spot U-Nets with 48-channel encoders and 96-channel decoders, the
/// first stage emitting 32 channels per frame triplet, Gaussian noise of σ = 30/255, and Adam at 1e-4 halved at
/// epochs 20, 25 and 30.
/// </para>
/// <para>
/// <b>For Beginners:</b> UDVD learns to denoise from noisy video alone. Its network is built so that it can never
/// see the pixel it is predicting, so the only way to predict that noisy pixel well is to predict the clean signal
/// from the surrounding pixels and frames; the noisy pixel is then folded back in with Bayes' rule.
/// </para>
/// </remarks>
public class UDVDOptions : ModelOptions
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
    public UDVDOptions(UDVDOptions other) : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderFeatures = other.EncoderFeatures;
        DecoderFeatures = other.DecoderFeatures;
        FirstStageFeatures = other.FirstStageFeatures;
        NoiseLevel = other.NoiseLevel;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
    }

    #region Architecture

    /// <summary>Gets or sets the channels of each blind-spot U-Net's encoder (48).</summary>
    public int EncoderFeatures { get; set; } = 48;

    /// <summary>Gets or sets the channels of each blind-spot U-Net's decoder and second stage's output (96).</summary>
    public int DecoderFeatures { get; set; } = 96;

    /// <summary>Gets or sets the channels the first-stage U-Net emits for each frame triplet (32).</summary>
    public int FirstStageFeatures { get; set; } = 32;

    /// <summary>
    /// Gets or sets the standard deviation (in normalized [0, 1] units) of the Gaussian noise the likelihood and the
    /// posterior assume; null estimates it from the frames. Default: 30/255, the level UDVD is trained at.
    /// </summary>
    public double? NoiseLevel { get; set; } = 30.0 / 255.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets Adam's starting learning rate (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    #endregion
}
