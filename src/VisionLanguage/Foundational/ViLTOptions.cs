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
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ViLTOptions(ViLTOptions other)
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
        PatchSize = other.PatchSize;
        UseWholeWordMasking = other.UseWholeWordMasking;
        UseRandAugment = other.UseRandAugment;
    }

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
