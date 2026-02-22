namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for OWLv2: self-training for scaling open-vocabulary detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OWLv2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class OWLv2Options : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OWLv2Options(OWLv2Options other)
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
        NumClassEmbeddings = other.NumClassEmbeddings;
        EnableSelfTraining = other.EnableSelfTraining;
    }

    public OWLv2Options()
    {
        VisionDim = 1024;
        DecoderDim = 1024;
        NumVisionLayers = 24;
        NumDecoderLayers = 6;
        NumHeads = 16;
        ImageSize = 960;
        MaxDetections = 100;
    }

    public int NumClassEmbeddings { get; set; } = 768;

    /// <summary>Gets or sets whether self-training augmentation is enabled.</summary>
    public bool EnableSelfTraining { get; set; } = true;
}
