using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Base configuration options for standalone vision encoder models.
/// </summary>
/// <remarks>
/// <para>
/// Vision encoders extract feature representations from images without a paired text encoder.
/// They are used as backbones for downstream VLMs, classification, detection, and segmentation tasks.
/// </para>
/// </remarks>
public class VisionEncoderOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VisionEncoderOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VisionEncoderOptions(VisionEncoderOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        EmbeddingDim = other.EmbeddingDim;
        PatchSize = other.PatchSize;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        FfnMultiplier = other.FfnMultiplier;
        DropoutRate = other.DropoutRate;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>
    /// Gets or sets the input image size (height = width).
    /// </summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>
    /// Gets or sets the vision encoder embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the ViT patch size in pixels.
    /// </summary>
    public int PatchSize { get; set; } = 16;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the feed-forward hidden dimension multiplier.
    /// </summary>
    public int FfnMultiplier { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the per-channel mean for image normalization.
    /// </summary>
    public double[] ImageMean { get; set; } = [0.485, 0.456, 0.406];

    /// <summary>
    /// Gets or sets the per-channel standard deviation for image normalization.
    /// </summary>
    public double[] ImageStd { get; set; } = [0.229, 0.224, 0.225];

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the initial learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 0.05;
}
