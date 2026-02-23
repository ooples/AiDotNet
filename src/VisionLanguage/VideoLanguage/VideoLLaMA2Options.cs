namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for VideoLLaMA 2: spatial-temporal convolution for video tokens.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VideoLLaMA2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VideoLLaMA2Options : VideoLanguageOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VideoLLaMA2Options(VideoLLaMA2Options other)
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
        EnableSpatialTemporalConv = other.EnableSpatialTemporalConv;
    }

    public VideoLLaMA2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "Mistral";
        MaxFrames = 16;
    }

    /// <summary>Gets or sets whether to use spatial-temporal convolution for video token compression.</summary>
    public bool EnableSpatialTemporalConv { get; set; } = true;
}
