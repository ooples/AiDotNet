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
