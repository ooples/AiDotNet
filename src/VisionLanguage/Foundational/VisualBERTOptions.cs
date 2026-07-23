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
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VisualBERTOptions(VisualBERTOptions other)
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
        MaxVisualTokens = other.MaxVisualTokens;
        UseVisualSegmentEmbeddings = other.UseVisualSegmentEmbeddings;
    }

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
