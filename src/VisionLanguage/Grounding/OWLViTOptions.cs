namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the OWLViT model. Default values follow the original paper settings.</para>
/// </remarks>
public class OWLViTOptions : GroundingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OWLViTOptions(OWLViTOptions other)
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
    }

    public OWLViTOptions()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 6;
        NumHeads = 12;
        ImageSize = 768;
        MaxDetections = 100;
    }

    /// <summary>Gets or sets the class embedding dimension.</summary>
    public int NumClassEmbeddings { get; set; } = 512;
}
