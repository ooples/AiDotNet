using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Base configuration options for unified understanding + generation vision models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UnifiedVision model. Default values follow the original paper settings.</para>
/// </remarks>
public class UnifiedVisionOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public UnifiedVisionOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UnifiedVisionOptions(UnifiedVisionOptions other)
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
        LanguageModelName = other.LanguageModelName;
        SupportsGeneration = other.SupportsGeneration;
        OutputImageSize = other.OutputImageSize;
        NumVisualTokens = other.NumVisualTokens;
    }

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets whether the model supports image generation.</summary>
    public bool SupportsGeneration { get; set; } = true;

    /// <summary>Gets or sets the generated image resolution.</summary>
    public int OutputImageSize { get; set; } = 512;

    /// <summary>Gets or sets the number of discrete visual tokens in the vocabulary.</summary>
    public int NumVisualTokens { get; set; } = 8192;
}
