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
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BridgeTowerOptions(BridgeTowerOptions other)
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
        NumBridgeLayers = other.NumBridgeLayers;
        BridgeDim = other.BridgeDim;
        UseBidirectionalBridges = other.UseBidirectionalBridges;
    }

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
