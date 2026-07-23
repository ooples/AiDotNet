using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Base configuration options for medical domain vision-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Medical model. Default values follow the original paper settings.</para>
/// </remarks>
public class MedicalVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MedicalVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedicalVLMOptions(MedicalVLMOptions other)
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

    /// <summary>Gets or sets the medical domain specialization.</summary>
    public string MedicalDomain { get; set; } = "General";

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the maximum output token length for report generation.</summary>
    public int MaxOutputTokens { get; set; } = 512;
}
