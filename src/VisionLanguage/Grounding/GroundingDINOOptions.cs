namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounding DINO: open-set detection combining DINO with grounded pre-training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GroundingDINO model. Default values follow the original paper settings.</para>
/// </remarks>
public class GroundingDINOOptions : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundingDINOOptions(GroundingDINOOptions other)
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
        NumQueryPositions = other.NumQueryPositions;
    }

    public GroundingDINOOptions()
    {
        VisionDim = 256;
        DecoderDim = 256;
        NumVisionLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        ImageSize = 800;
        MaxDetections = 300;
    }

    /// <summary>Gets or sets the number of object query positions.</summary>
    public int NumQueryPositions { get; set; } = 900;
}
