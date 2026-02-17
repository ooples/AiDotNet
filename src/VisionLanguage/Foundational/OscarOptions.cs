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
        VisionDim = 2054;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
    }
}
