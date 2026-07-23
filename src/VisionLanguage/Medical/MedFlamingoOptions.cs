using AiDotNet.VisionLanguage.Medical;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Configuration options for Med-Flamingo.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MedFlamingo model. Default values follow the original paper settings.</para>
/// </remarks>
public class MedFlamingoOptions : MedicalVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedFlamingoOptions(MedFlamingoOptions other)
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
        MaxFewShotExamples = other.MaxFewShotExamples;
    }

    public MedFlamingoOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "MPT";
        MedicalDomain = "General";
    }

    /// <summary>Gets or sets the number of few-shot examples supported.</summary>
    public int MaxFewShotExamples { get; set; } = 8;
}
