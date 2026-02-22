namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for VideoLLaMA 3: frontier multimodal for image and video.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VideoLLaMA3 model. Default values follow the original paper settings.</para>
/// </remarks>
public class VideoLLaMA3Options : VideoLanguageOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VideoLLaMA3Options(VideoLLaMA3Options other)
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
    }

    public VideoLLaMA3Options()
    {
        VisionDim = 1152;
        DecoderDim = 4096;
        NumVisionLayers = 27;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 128256;
        LanguageModelName = "LLaMA-3";
        MaxFrames = 128;
    }
}
