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
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ViLBERTOptions(ViLBERTOptions other)
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
        NumCoAttentionLayers = other.NumCoAttentionLayers;
        MaxVisualRegions = other.MaxVisualRegions;
    }

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
