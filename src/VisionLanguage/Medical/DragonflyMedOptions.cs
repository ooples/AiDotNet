using AiDotNet.VisionLanguage.Medical;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Configuration options for Dragonfly-Med.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DragonflyMed model. Default values follow the original paper settings.</para>
/// </remarks>
public class DragonflyMedOptions : MedicalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DragonflyMedOptions(DragonflyMedOptions other)
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
        MedicalDomain = other.MedicalDomain;
        LanguageModelName = other.LanguageModelName;
        MaxOutputTokens = other.MaxOutputTokens;
    }

    public DragonflyMedOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        MedicalDomain = "General";
    }
}
