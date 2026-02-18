using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for ViLT (Vision-and-Language Transformer).
/// </summary>
/// <remarks>
/// <para>ViLT (Kim et al., ICML 2021) is a minimal architecture that removes the CNN/object detector
/// entirely. Raw image patches are linearly embedded and concatenated with text tokens in a single
/// transformer, making it 60x faster than region-feature-based models at comparable accuracy.</para>
/// </remarks>
public class ViLTOptions : FoundationalVLMOptions
{
    /// <summary>Gets or sets the patch size for image tokenization.</summary>
    public int PatchSize { get; set; } = 32;

    /// <summary>Gets or sets whether to use whole-word masking for MLM pre-training.</summary>
    public bool UseWholeWordMasking { get; set; } = true;

    /// <summary>Gets or sets whether to use image augmentation during training.</summary>
    public bool UseRandAugment { get; set; } = true;

    public ViLTOptions()
    {
        FusionType = FusionType.SingleStream;
        VisualFeatureType = VisualFeatureType.PatchEmbeddings;
        VisionDim = 768;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
        ImageSize = 384;
    }
}
