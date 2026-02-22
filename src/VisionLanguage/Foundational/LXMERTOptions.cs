using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for LXMERT cross-modal encoder.
/// </summary>
/// <remarks>
/// <para>LXMERT (Tan and Bansal, EMNLP 2019) has three encoder types: object relationship encoder,
/// language encoder, and cross-modality encoder with cross-attention layers.</para>
/// </remarks>
public class LXMERTOptions : FoundationalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LXMERTOptions(LXMERTOptions other)
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
        NumRelationshipLayers = other.NumRelationshipLayers;
        NumCrossModalityLayers = other.NumCrossModalityLayers;
        MaxVisualObjects = other.MaxVisualObjects;
    }

    /// <summary>Gets or sets the number of object relationship encoder layers.</summary>
    public int NumRelationshipLayers { get; set; } = 5;

    /// <summary>Gets or sets the number of cross-modality encoder layers.</summary>
    public int NumCrossModalityLayers { get; set; } = 5;

    /// <summary>Gets or sets the maximum number of visual objects.</summary>
    public int MaxVisualObjects { get; set; } = 36;

    public LXMERTOptions()
    {
        FusionType = FusionType.CrossModal;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 2048;
        TextDim = 768;
        FusionDim = 768;
        NumTextLayers = 9;
        NumFusionLayers = 5;
    }
}
