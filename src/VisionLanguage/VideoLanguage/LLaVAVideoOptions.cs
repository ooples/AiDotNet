namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for LLaVA-Video: synthetic dataset-trained video instruction model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the LLaVAVideo model. Default values follow the original paper settings.</para>
/// </remarks>
public class LLaVAVideoOptions : VideoLanguageOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LLaVAVideoOptions(LLaVAVideoOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        ImageChannels = other.ImageChannels;
        PatchSize = other.PatchSize;
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
        MaxFrames = other.MaxFrames;
        LanguageModelName = other.LanguageModelName;
        ProjectionDim = other.ProjectionDim;
        SystemPrompt = other.SystemPrompt;
    }

    public LLaVAVideoOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        ImageChannels = 3;
        PatchSize = 16;
        VocabSize = 32000;
        LanguageModelName = "Qwen2";
        MaxFrames = 64;
    }

    /// <summary>
    /// Number of channels per video frame. Default: 3 (RGB) per the LLaVA-Video paper.
    /// </summary>
    public int ImageChannels { get; set; }

    /// <summary>
    /// Patch size for the vision encoder. Default: 16 (paper uses CLIP ViT/16 for the original config).
    /// </summary>
    public int PatchSize { get; set; }
}
