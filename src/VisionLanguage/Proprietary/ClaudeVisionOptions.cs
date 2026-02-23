using AiDotNet.VisionLanguage.Proprietary;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Configuration options for Claude Vision.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ClaudeVision model. Default values follow the original paper settings.</para>
/// </remarks>
public class ClaudeVisionOptions : ProprietaryVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ClaudeVisionOptions(ClaudeVisionOptions other)
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
        ExtendedThinking = other.ExtendedThinking;
    }

    public ClaudeVisionOptions()
    {
        VisionDim = 1024;
        DecoderDim = 8192;
        NumVisionLayers = 32;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 448;
        VocabSize = 100000;
        Provider = "Anthropic";
        LanguageModelName = "Claude";
        MaxContextLength = 200000;
    }

    /// <summary>Gets or sets whether extended thinking mode is enabled.</summary>
    public bool ExtendedThinking { get; set; } = true;
}
