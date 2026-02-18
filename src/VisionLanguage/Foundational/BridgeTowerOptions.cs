using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for BridgeTower cross-modal alignment model.
/// </summary>
/// <remarks>
/// <para>BridgeTower (Xu et al., AAAI 2023) introduces bridge layers that connect vision and text
/// encoder layers at multiple levels, enabling fine-grained cross-modal alignment. Each bridge
/// layer consists of cross-attention between corresponding encoder layers.</para>
/// </remarks>
public class BridgeTowerOptions : FoundationalVLMOptions
{
    /// <summary>Gets or sets the number of bridge connection layers.</summary>
    public int NumBridgeLayers { get; set; } = 6;

    /// <summary>Gets or sets the bridge layer hidden dimension.</summary>
    public int BridgeDim { get; set; } = 768;

    /// <summary>Gets or sets whether to use bi-directional bridges.</summary>
    public bool UseBidirectionalBridges { get; set; } = true;

    public BridgeTowerOptions()
    {
        FusionType = FusionType.BridgeLayers;
        VisualFeatureType = VisualFeatureType.PatchEmbeddings;
        VisionDim = 768;
        TextDim = 768;
        FusionDim = 768;
        NumVisionLayers = 12;
        NumTextLayers = 12;
        NumFusionLayers = 6;
    }
}
