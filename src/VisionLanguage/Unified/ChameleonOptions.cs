namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Chameleon: early fusion with discrete tokens for all modalities.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Chameleon model. Default values follow the original paper settings.</para>
/// </remarks>
public class ChameleonOptions : UnifiedVisionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ChameleonOptions(ChameleonOptions other)
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

    public ChameleonOptions()
    {
        VisionDim = 4096;
        DecoderDim = 4096;
        NumVisionLayers = 0;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 65536;
        LanguageModelName = "Chameleon";
        NumVisualTokens = 8192;
    }
}
