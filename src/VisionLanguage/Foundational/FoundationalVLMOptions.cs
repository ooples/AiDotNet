using AiDotNet.Models.Options;
using AiDotNet.Onnx;
using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Base configuration options for foundational vision-language fusion models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Foundational model. Default values follow the original paper settings.</para>
/// </remarks>
public class FoundationalVLMOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public FoundationalVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FoundationalVLMOptions(FoundationalVLMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        TextDim = other.TextDim;
        FusionDim = other.FusionDim;
        NumVisionLayers = other.NumVisionLayers;
        NumTextLayers = other.NumTextLayers;
        NumFusionLayers = other.NumFusionLayers;
        NumHeads = other.NumHeads;
        MaxSequenceLength = other.MaxSequenceLength;
        VocabSize = other.VocabSize;
        DropoutRate = other.DropoutRate;
        FusionType = other.FusionType;
        VisualFeatureType = other.VisualFeatureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets the input image size.</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>Gets or sets the vision feature dimension.</summary>
    public int VisionDim { get; set; } = 768;

    /// <summary>Gets or sets the text embedding dimension.</summary>
    public int TextDim { get; set; } = 768;

    /// <summary>Gets or sets the fusion/hidden dimension.</summary>
    public int FusionDim { get; set; } = 768;

    /// <summary>Gets or sets the number of vision encoder layers.</summary>
    public int NumVisionLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of text encoder layers.</summary>
    public int NumTextLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of cross-modal/fusion layers.</summary>
    public int NumFusionLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the maximum text sequence length.</summary>
    public int MaxSequenceLength { get; set; } = 128;

    /// <summary>Gets or sets the vocabulary size.</summary>
    public int VocabSize { get; set; } = 30522;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the fusion type.</summary>
    public FusionType FusionType { get; set; } = FusionType.DualStream;

    /// <summary>Gets or sets the visual feature type.</summary>
    public VisualFeatureType VisualFeatureType { get; set; } = VisualFeatureType.RegionFeatures;

    /// <summary>Gets or sets the per-channel image normalization mean.</summary>
    public double[] ImageMean { get; set; } = [0.485, 0.456, 0.406];

    /// <summary>Gets or sets the per-channel image normalization std.</summary>
    public double[] ImageStd { get; set; } = [0.229, 0.224, 0.225];

    /// <summary>Gets or sets the ONNX model path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
