using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for PaLM-E.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PaLME model. Default values follow the original paper settings.</para>
/// </remarks>
public class PaLMEOptions : VisionLanguageActionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PaLMEOptions(PaLMEOptions other)
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
        TotalParameters = other.TotalParameters;
    }

    public PaLMEOptions()
    {
        VisionDim = 1408;
        DecoderDim = 8192;
        NumVisionLayers = 48;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 224;
        VocabSize = 256000;
        LanguageModelName = "PaLM";
        ActionDimension = 7;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; } = 562;
}
