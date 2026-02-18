using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for METER (Multimodal End-to-end TransformER).
/// </summary>
/// <remarks>
/// <para>METER (Dou et al., CVPR 2022) is a systematic study of vision-language pre-training
/// components. It uses separate CLIP ViT vision encoder and RoBERTa text encoder, connected
/// by a co-attention transformer fusion module, providing an optimized combination of
/// architecture choices for VLP.</para>
/// </remarks>
public class METEROptions : FoundationalVLMOptions
{
    /// <summary>Gets or sets the number of cross-attention fusion layers.</summary>
    public int NumCrossAttentionLayers { get; set; } = 6;

    /// <summary>Gets or sets the vision encoder type.</summary>
    public ViTVariant VisionEncoder { get; set; } = ViTVariant.ViTB16;

    /// <summary>Gets or sets the text encoder type.</summary>
    public TextEncoderVariant TextEncoder { get; set; } = TextEncoderVariant.RoBERTa;

    public METEROptions()
    {
        FusionType = FusionType.CoAttention;
        VisualFeatureType = VisualFeatureType.PatchEmbeddings;
        VisionDim = 768;
        TextDim = 768;
        FusionDim = 768;
        NumVisionLayers = 12;
        NumTextLayers = 12;
        NumFusionLayers = 6;
    }
}
