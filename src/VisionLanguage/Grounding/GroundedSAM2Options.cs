namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounded-SAM 2: combines Grounding DINO with SAM 2 for grounded segmentation and tracking.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GroundedSAM2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class GroundedSAM2Options : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundedSAM2Options(GroundedSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        MaxDetections = other.MaxDetections;
        ConfidenceThreshold = other.ConfidenceThreshold;
        NmsThreshold = other.NmsThreshold;
        BoxDimension = other.BoxDimension;
        EnableSegmentation = other.EnableSegmentation;
        EnableTracking = other.EnableTracking;
    }

    public GroundedSAM2Options()
    {
        VisionDim = 256;
        DecoderDim = 256;
        NumVisionLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        ImageSize = 1024;
        MaxDetections = 300;
    }

    /// <summary>Gets or sets whether to produce segmentation masks.</summary>
    public bool EnableSegmentation { get; set; } = true;

    /// <summary>Gets or sets whether to enable video object tracking.</summary>
    public bool EnableTracking { get; set; } = true;
}
