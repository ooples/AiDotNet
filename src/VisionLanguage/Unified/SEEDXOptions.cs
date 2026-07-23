namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for SEED-X: multi-granularity comprehension and generation model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SEEDX model. Default values follow the original paper settings.</para>
/// </remarks>
public class SEEDXOptions : UnifiedVisionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SEEDXOptions(SEEDXOptions other)
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
        EnableMultiGranularity = other.EnableMultiGranularity;
    }

    public SEEDXOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        NumVisualTokens = 8192;
    }

    /// <summary>Gets or sets whether to use multi-granularity visual encoding.</summary>
    public bool EnableMultiGranularity { get; set; } = true;
}
