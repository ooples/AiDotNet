using AiDotNet.VisionLanguage.Proprietary;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Configuration options for Gemini Vision.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GeminiVision model. Default values follow the original paper settings.</para>
/// </remarks>
public class GeminiVisionOptions : ProprietaryVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GeminiVisionOptions(GeminiVisionOptions other)
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
        MaxContextTokens = other.MaxContextTokens;
        NumExperts = other.NumExperts;
    }

    public GeminiVisionOptions()
    {
        VisionDim = 1024;
        DecoderDim = 8192;
        NumVisionLayers = 32;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 448;
        VocabSize = 256000;
        Provider = "Google";
        LanguageModelName = "Gemini-MoE";
        MaxContextLength = 2000000;
    }

    /// <summary>Gets or sets the maximum context length in tokens.</summary>
    public int MaxContextTokens { get; set; } = 2000000;

    /// <summary>Gets or sets the number of MoE experts.</summary>
    public int NumExperts { get; set; } = 16;
}
