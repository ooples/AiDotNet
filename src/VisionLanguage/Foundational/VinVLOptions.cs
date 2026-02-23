using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for VinVL (Visual Features in Vision-Language).
/// </summary>
/// <remarks>
/// <para>VinVL (Zhang et al., CVPR 2021) improves on Oscar by providing better visual features
/// through a stronger object detection backbone (ResNeXt-152 C4), enriching object tags with
/// attributes and achieving state-of-the-art on VQA, captioning, and retrieval benchmarks.</para>
/// </remarks>
public class VinVLOptions : FoundationalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VinVLOptions(VinVLOptions other)
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
        UseAttributePredictions = other.UseAttributePredictions;
        DetectionThreshold = other.DetectionThreshold;
    }

    /// <summary>Gets or sets the maximum number of object tags with attributes.</summary>
    public int MaxObjectTags { get; set; } = 50;

    /// <summary>Gets or sets the maximum number of image regions.</summary>
    public int MaxImageRegions { get; set; } = 50;

    /// <summary>Gets or sets whether to include attribute predictions in object tags.</summary>
    public bool UseAttributePredictions { get; set; } = true;

    /// <summary>Gets or sets the object detection confidence threshold.</summary>
    public double DetectionThreshold { get; set; } = 0.2;

    public VinVLOptions()
    {
        FusionType = FusionType.SingleStream;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 2048;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
    }
}
