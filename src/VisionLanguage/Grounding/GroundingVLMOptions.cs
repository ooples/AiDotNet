using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Base configuration options for visual grounding models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Grounding model. Default values follow the original paper settings.</para>
/// </remarks>
public class GroundingVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GroundingVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundingVLMOptions(GroundingVLMOptions other)
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
    }

    /// <summary>Gets or sets the maximum number of detections per image.</summary>
    public int MaxDetections { get; set; } = 300;

    /// <summary>Gets or sets the confidence threshold for detection filtering.</summary>
    public double ConfidenceThreshold { get; set; } = 0.25;

    /// <summary>Gets or sets the IoU threshold for non-maximum suppression.</summary>
    public double NmsThreshold { get; set; } = 0.5;

    /// <summary>Gets or sets the number of output coordinates per box (typically 4 for [x1,y1,x2,y2]).</summary>
    public int BoxDimension { get; set; } = 4;
}
