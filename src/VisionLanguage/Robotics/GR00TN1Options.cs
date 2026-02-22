using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for GR00T N1.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GR00TN1 model. Default values follow the original paper settings.</para>
/// </remarks>
public class GR00TN1Options : VisionLanguageActionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GR00TN1Options(GR00TN1Options other)
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
        ActionDimension = other.ActionDimension;
        LanguageModelName = other.LanguageModelName;
        PredictionHorizon = other.PredictionHorizon;
        ObservationHistory = other.ObservationHistory;
        NumJoints = other.NumJoints;
    }

    public GR00TN1Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "Eagle";
        ActionDimension = 52;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the number of humanoid joints controlled.</summary>
    public int NumJoints { get; set; } = 52;
}
