namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Transfusion: combined autoregressive and diffusion loss in single transformer.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Transfusion model. Default values follow the original paper settings.</para>
/// </remarks>
public class TransfusionOptions : UnifiedVisionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TransfusionOptions(TransfusionOptions other)
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
        EnableDiffusionLoss = other.EnableDiffusionLoss;
    }

    public TransfusionOptions()
    {
        VisionDim = 4096;
        DecoderDim = 4096;
        NumVisionLayers = 0;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 256;
        VocabSize = 32000;
        LanguageModelName = "Transfusion";
        NumVisualTokens = 256;
    }

    /// <summary>Gets or sets whether to use diffusion loss for image generation.</summary>
    public bool EnableDiffusionLoss { get; set; } = true;
}
