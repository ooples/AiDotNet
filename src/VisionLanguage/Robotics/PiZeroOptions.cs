using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for pi-zero.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PiZero model. Default values follow the original paper settings.</para>
/// </remarks>
public class PiZeroOptions : VisionLanguageActionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PiZeroOptions(PiZeroOptions other)
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
        NumFlowSteps = other.NumFlowSteps;
    }

    public PiZeroOptions()
    {
        VisionDim = 1152;
        DecoderDim = 2048;
        NumVisionLayers = 27;
        NumDecoderLayers = 18;
        NumHeads = 16;
        ImageSize = 224;
        VocabSize = 256000;
        LanguageModelName = "PaliGemma";
        ActionDimension = 7;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the number of flow matching denoising steps.</summary>
    public int NumFlowSteps { get; set; } = 10;
}
