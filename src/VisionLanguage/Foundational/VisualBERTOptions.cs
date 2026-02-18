using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for VisualBERT single-stream fusion model.
/// </summary>
/// <remarks>
/// <para>VisualBERT (Li et al., 2019) concatenates visual tokens (from Faster R-CNN) with text tokens
/// in a single BERT transformer stream, allowing implicit cross-modal alignment through self-attention.</para>
/// </remarks>
public class VisualBERTOptions : FoundationalVLMOptions
{
    /// <summary>Gets or sets the maximum number of visual tokens.</summary>
    public int MaxVisualTokens { get; set; } = 36;

    /// <summary>Gets or sets whether to use visual segment embeddings.</summary>
    public bool UseVisualSegmentEmbeddings { get; set; } = true;

    public VisualBERTOptions()
    {
        FusionType = FusionType.SingleStream;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 2048;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
    }
}
