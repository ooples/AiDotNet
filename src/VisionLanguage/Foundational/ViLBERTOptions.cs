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
        VisualFeatureDim = other.VisualFeatureDim;
        TextDim = other.TextDim;
        FusionDim = other.FusionDim;
        NumVisionLayers = other.NumVisionLayers;
        NumTextLayers = other.NumTextLayers;
        NumFusionLayers = other.NumFusionLayers;
        NumHeads = other.NumHeads;
        NumVisionHeads = other.NumVisionHeads;
        NumTextHeads = other.NumTextHeads;
        NumFusionHeads = other.NumFusionHeads;
        VisionIntermediateDim = other.VisionIntermediateDim;
        TextIntermediateDim = other.TextIntermediateDim;
        FusionIntermediateDim = other.FusionIntermediateDim;
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
    /// <remarks>This is an alias for <see cref="FoundationalVLMOptions.NumFusionLayers"/>.</remarks>
    public int NumCoAttentionLayers
    {
        get => NumFusionLayers;
        set => NumFusionLayers = value;
    }

    /// <summary>Gets or sets the maximum number of visual regions (object proposals).</summary>
    public int MaxVisualRegions { get; set; } = 36;

    /// <summary>Gets or sets the Faster R-CNN detector feature width before visual embedding.</summary>
    public int VisualFeatureDim { get; set; } = 2048;

    /// <summary>Gets or sets the number of self-attention heads in the visual stream.</summary>
    public int NumVisionHeads { get; set; } = 8;

    /// <summary>Gets or sets the number of self-attention heads in the BERT text stream.</summary>
    public int NumTextHeads { get; set; } = 12;

    /// <summary>Gets or sets the number of heads in the bidirectional co-attention blocks.</summary>
    public int NumFusionHeads { get; set; } = 8;

    /// <summary>Gets or sets the visual-stream feed-forward width.</summary>
    public int VisionIntermediateDim { get; set; } = 1024;

    /// <summary>Gets or sets the BERT text-stream feed-forward width.</summary>
    public int TextIntermediateDim { get; set; } = 3072;

    /// <summary>Gets or sets the co-attention feed-forward width.</summary>
    public int FusionIntermediateDim { get; set; } = 1024;

    public ViLBERTOptions()
    {
        FusionType = FusionType.CoAttention;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 1024;
        TextDim = 768;
        FusionDim = 1024;
        NumVisionLayers = 6;
        NumTextLayers = 12;
        NumFusionLayers = 6;
        NumHeads = 8; // Backward-compatible aggregate; stream-specific defaults are exposed above.
        MaxSequenceLength = 512;
    }
}
