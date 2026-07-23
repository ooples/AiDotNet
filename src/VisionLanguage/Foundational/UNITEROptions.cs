using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for UNITER (Universal Image-TExt Representation).
/// </summary>
/// <remarks>
/// <para>UNITER (Chen et al., ECCV 2020) uses conditional masking pre-training where either
/// image regions or text tokens are masked, forcing the model to learn cross-modal alignment.
/// It uses a single-stream transformer for joint image-text encoding.</para>
/// </remarks>
public class UNITEROptions : FoundationalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UNITEROptions(UNITEROptions other)
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
        ImageMaskProbability = other.ImageMaskProbability;
        TextMaskProbability = other.TextMaskProbability;
        MaxImageRegions = other.MaxImageRegions;
    }

    /// <summary>Gets or sets the image region masking probability during training.</summary>
    public double ImageMaskProbability { get; set; } = 0.15;

    /// <summary>Gets or sets the text token masking probability during training.</summary>
    public double TextMaskProbability { get; set; } = 0.15;

    /// <summary>Gets or sets the maximum number of image regions.</summary>
    public int MaxImageRegions { get; set; } = 36;

    public UNITEROptions()
    {
        FusionType = FusionType.SingleStream;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 2048;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
    }
}
