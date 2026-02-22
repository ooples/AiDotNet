using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Base configuration options for contrastive vision-language encoders (CLIP-family models).
/// </summary>
/// <remarks>
/// <para>
/// Contrastive encoders learn a shared embedding space for images and text via contrastive learning.
/// They share common hyperparameters: image/text embedding dimensions, projection dimension,
/// temperature, and image preprocessing settings.
/// </para>
/// <para>
/// <b>For Beginners:</b> These settings control how the model processes images and text.
/// The most important ones are:
/// <list type="bullet">
/// <item><b>ImageSize</b>: The size images are resized to before processing (e.g., 224x224 pixels)</item>
/// <item><b>EmbeddingDim</b>: How many numbers represent each image or text (bigger = more detail)</item>
/// <item><b>Temperature</b>: How "sharp" the similarity comparisons are (lower = more decisive)</item>
/// </list>
/// </para>
/// </remarks>
public class ContrastiveEncoderOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ContrastiveEncoderOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ContrastiveEncoderOptions(ContrastiveEncoderOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionEmbeddingDim = other.VisionEmbeddingDim;
        VisionEncoderVariant = other.VisionEncoderVariant;
        PatchSize = other.PatchSize;
        NumVisionLayers = other.NumVisionLayers;
        NumVisionHeads = other.NumVisionHeads;
        VisionFfnMultiplier = other.VisionFfnMultiplier;
        TextEmbeddingDim = other.TextEmbeddingDim;
        TextEncoderVariant = other.TextEncoderVariant;
        MaxSequenceLength = other.MaxSequenceLength;
        VocabSize = other.VocabSize;
        NumTextLayers = other.NumTextLayers;
        NumTextHeads = other.NumTextHeads;
        ProjectionDim = other.ProjectionDim;
        Temperature = other.Temperature;
        DropoutRate = other.DropoutRate;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ImageEncoderModelPath = other.ImageEncoderModelPath;
        TextEncoderModelPath = other.TextEncoderModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        WarmUpSteps = other.WarmUpSteps;
        LabelSmoothing = other.LabelSmoothing;
    }

    #region Image Encoder

    /// <summary>
    /// Gets or sets the input image size (height = width).
    /// </summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>
    /// Gets or sets the vision encoder embedding dimension.
    /// </summary>
    public int VisionEmbeddingDim { get; set; } = 768;

    /// <summary>
    /// Gets or sets the vision encoder variant.
    /// </summary>
    public ViTVariant VisionEncoderVariant { get; set; } = ViTVariant.ViTB32;

    /// <summary>
    /// Gets or sets the ViT patch size in pixels.
    /// </summary>
    public int PatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of vision transformer layers.
    /// </summary>
    public int NumVisionLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of vision attention heads.
    /// </summary>
    public int NumVisionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the feed-forward hidden dimension multiplier for vision encoder.
    /// </summary>
    public int VisionFfnMultiplier { get; set; } = 4;

    #endregion

    #region Text Encoder

    /// <summary>
    /// Gets or sets the text encoder embedding dimension.
    /// </summary>
    public int TextEmbeddingDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the text encoder variant.
    /// </summary>
    public TextEncoderVariant TextEncoderVariant { get; set; } = TextEncoderVariant.Transformer;

    /// <summary>
    /// Gets or sets the maximum text token sequence length.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 77;

    /// <summary>
    /// Gets or sets the vocabulary size.
    /// </summary>
    public int VocabSize { get; set; } = 49408;

    /// <summary>
    /// Gets or sets the number of text transformer layers.
    /// </summary>
    public int NumTextLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of text attention heads.
    /// </summary>
    public int NumTextHeads { get; set; } = 8;

    #endregion

    #region Projection

    /// <summary>
    /// Gets or sets the shared projection dimension for the joint embedding space.
    /// </summary>
    public int ProjectionDim { get; set; } = 512;

    #endregion

    #region Contrastive Learning

    /// <summary>
    /// Gets or sets the temperature parameter for contrastive loss.
    /// </summary>
    /// <remarks>
    /// <para>Controls the sharpness of the similarity distribution. CLIP uses a learnable temperature
    /// initialized at log(1/0.07). Lower values make the model more discriminative.</para>
    /// </remarks>
    public double Temperature { get; set; } = 0.07;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion

    #region Image Preprocessing

    /// <summary>
    /// Gets or sets the per-channel mean for image normalization.
    /// </summary>
    public double[] ImageMean { get; set; } = [0.48145466, 0.4578275, 0.40821073];

    /// <summary>
    /// Gets or sets the per-channel standard deviation for image normalization.
    /// </summary>
    public double[] ImageStd { get; set; } = [0.26862954, 0.26130258, 0.27577711];

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to the image encoder ONNX model.
    /// </summary>
    public string? ImageEncoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the text encoder ONNX model.
    /// </summary>
    public string? TextEncoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the initial learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the warm-up steps for the learning rate scheduler.
    /// </summary>
    public int WarmUpSteps { get; set; } = 10000;

    /// <summary>
    /// Gets or sets the label smoothing factor.
    /// </summary>
    public double LabelSmoothing { get; set; } = 0.0;

    #endregion
}
