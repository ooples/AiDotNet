using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Base configuration options for video-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VideoLanguage model. Default values follow the original paper settings.</para>
/// </remarks>
public class VideoLanguageOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VideoLanguageOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VideoLanguageOptions(VideoLanguageOptions other)
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

    /// <summary>Gets or sets the maximum number of video frames the model can process.</summary>
    public int MaxFrames { get; set; } = 32;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the MLP projection hidden dimension.</summary>
    public int ProjectionDim { get; set; } = 4096;

    /// <summary>Gets or sets the system prompt for chat mode.</summary>
    public string SystemPrompt { get; set; } = "You are a helpful video understanding assistant.";
}
