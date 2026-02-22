using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Base configuration options for proprietary VLM reference implementations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Proprietary model. Default values follow the original paper settings.</para>
/// </remarks>
public class ProprietaryVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ProprietaryVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ProprietaryVLMOptions(ProprietaryVLMOptions other)
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
        Provider = other.Provider;
        LanguageModelName = other.LanguageModelName;
        MaxContextLength = other.MaxContextLength;
    }

    /// <summary>Gets or sets the proprietary model provider name.</summary>
    public string Provider { get; set; } = "Unknown";

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "Proprietary";

    /// <summary>Gets or sets the maximum context window length in tokens.</summary>
    public int MaxContextLength { get; set; } = 128000;
}
