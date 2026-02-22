using AiDotNet.VisionLanguage.Medical;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Configuration options for PathVLM.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Path model. Default values follow the original paper settings.</para>
/// </remarks>
public class PathVLMOptions : MedicalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PathVLMOptions(PathVLMOptions other)
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
        MagnificationLevel = other.MagnificationLevel;
    }

    public PathVLMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        MedicalDomain = "Pathology";
    }

    /// <summary>Gets or sets the pathology image magnification level.</summary>
    public int MagnificationLevel { get; set; } = 20;
}
