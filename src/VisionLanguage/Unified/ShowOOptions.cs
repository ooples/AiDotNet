namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Show-o: single transformer for unified understanding and generation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ShowO model. Default values follow the original paper settings.</para>
/// </remarks>
public class ShowOOptions : UnifiedVisionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ShowOOptions(ShowOOptions other)
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

    public ShowOOptions()
    {
        VisionDim = 1024;
        DecoderDim = 2048;
        NumVisionLayers = 24;
        NumDecoderLayers = 24;
        NumHeads = 16;
        ImageSize = 256;
        VocabSize = 32000;
        LanguageModelName = "Phi-1.5";
        NumVisualTokens = 8192;
    }
}
