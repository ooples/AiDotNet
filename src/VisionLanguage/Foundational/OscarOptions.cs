using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for Oscar (Object-Semantics Aligned pre-training).
/// </summary>
/// <remarks>
/// <para>Oscar (Li et al., ECCV 2020) uses detected object tags as "anchor points" to align image
/// regions with text tokens, forming triples of (word tokens, object tags, region features)
/// that are fed into a single BERT transformer.</para>
/// </remarks>
public class OscarOptions : FoundationalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OscarOptions(OscarOptions other)
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
        MaxObjectTags = other.MaxObjectTags;
        MaxImageRegions = other.MaxImageRegions;
        ContrastiveLossWeight = other.ContrastiveLossWeight;
    }

    /// <summary>Gets or sets the maximum number of object tags per image.</summary>
    public int MaxObjectTags { get; set; } = 50;

    /// <summary>Gets or sets the maximum number of image regions.</summary>
    public int MaxImageRegions { get; set; } = 50;

    /// <summary>Gets or sets the contrastive loss weight for tag-text alignment.</summary>
    public double ContrastiveLossWeight { get; set; } = 1.0;

    public OscarOptions()
    {
        FusionType = FusionType.SingleStream;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 2048;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
    }
}
