namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Ferret-v2: improved referring and grounding with enhanced spatial understanding.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FerretV2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class FerretV2Options : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FerretV2Options(FerretV2Options other)
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
        EnableFreeFormRegions = other.EnableFreeFormRegions;
        EnableHighResolution = other.EnableHighResolution;
    }

    public FerretV2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        MaxDetections = 100;
        VocabSize = 32000;
    }

    public bool EnableFreeFormRegions { get; set; } = true;

    /// <summary>Gets or sets whether to use high-resolution input processing.</summary>
    public bool EnableHighResolution { get; set; } = true;
}
