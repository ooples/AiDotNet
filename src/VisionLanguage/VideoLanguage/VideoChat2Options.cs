namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for VideoChat2: progressive video training with diverse data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VideoChat2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VideoChat2Options : VideoLanguageOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VideoChat2Options(VideoChat2Options other)
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
        MaxFrames = other.MaxFrames;
        LanguageModelName = other.LanguageModelName;
        ProjectionDim = other.ProjectionDim;
        SystemPrompt = other.SystemPrompt;
        QFormerDim = other.QFormerDim;
        NumQFormerLayers = other.NumQFormerLayers;
        NumQueryTokens = other.NumQueryTokens;
        NumQFormerHeads = other.NumQFormerHeads;
    }

    public VideoChat2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "Mistral";
        MaxFrames = 16;
    }

    /// <summary>Q-Former hidden width (BLIP-2 default 768). VideoChat2 (Li et al. 2023,
    /// arXiv:2311.17005) resamples the video features with a Q-Former before the LLM.</summary>
    public int QFormerDim { get; set; } = 768;

    /// <summary>Number of Q-Former transformer blocks (BLIP-2 default 12).</summary>
    public int NumQFormerLayers { get; set; } = 12;

    /// <summary>Number of learnable Q-Former query tokens that cross-attend to the video features
    /// (BLIP-2 default 32).</summary>
    public int NumQueryTokens { get; set; } = 32;

    /// <summary>Number of attention heads inside the Q-Former (BLIP-2 default 12).</summary>
    public int NumQFormerHeads { get; set; } = 12;
}
