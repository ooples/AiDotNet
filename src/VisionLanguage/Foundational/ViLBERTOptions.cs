using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for ViLBERT (Vision-and-Language BERT) with co-attention.
/// </summary>
/// <remarks>
/// <para>ViLBERT (Lu et al., NeurIPS 2019) processes images and text in two parallel BERT-like streams
/// connected by co-attention transformer layers, enabling rich cross-modal interaction.</para>
/// </remarks>
public class ViLBERTOptions : FoundationalVLMOptions
{
    /// <summary>Gets or sets the number of co-attention layers between streams.</summary>
    public int NumCoAttentionLayers { get; set; } = 6;

    /// <summary>Gets or sets the maximum number of visual regions (object proposals).</summary>
    public int MaxVisualRegions { get; set; } = 36;

    public ViLBERTOptions()
    {
        FusionType = FusionType.CoAttention;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 1024;
        TextDim = 768;
        FusionDim = 1024;
        NumFusionLayers = 6;
    }
}
