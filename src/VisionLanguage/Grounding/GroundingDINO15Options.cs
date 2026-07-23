namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounding DINO 1.5: enhanced open-set detection with improved architecture.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GroundingDINO15 model. Default values follow the original paper settings.</para>
/// </remarks>
public class GroundingDINO15Options : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundingDINO15Options(GroundingDINO15Options other)
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
        BackboneType = other.BackboneType;
    }

    public GroundingDINO15Options()
    {
        VisionDim = 256;
        DecoderDim = 256;
        NumVisionLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        ImageSize = 800;
        MaxDetections = 300;
    }

    public int NumQueryPositions { get; set; } = 900;

    /// <summary>Gets or sets the visual backbone type.</summary>
    public string BackboneType { get; set; } = "Swin-L";
}
