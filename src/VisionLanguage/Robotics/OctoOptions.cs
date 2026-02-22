using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for Octo.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Octo model. Default values follow the original paper settings.</para>
/// </remarks>
public class OctoOptions : VisionLanguageActionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public OctoOptions(OctoOptions other)
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

    public OctoOptions()
    {
        VisionDim = 384;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 256;
        VocabSize = 32000;
        LanguageModelName = "Transformer";
        ActionDimension = 7;
        PredictionHorizon = 4;
        ObservationHistory = 2;
    }

    /// <summary>Gets or sets the total parameter count in millions.</summary>
    public int TotalParameters { get; set; } = 93;
}
