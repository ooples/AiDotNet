namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for SlowFast-LLaVA: token-efficient slow/fast pathways for long video.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SlowFastLLaVA model. Default values follow the original paper settings.</para>
/// </remarks>
public class SlowFastLLaVAOptions : VideoLanguageOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SlowFastLLaVAOptions(SlowFastLLaVAOptions other)
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
        SlowFrames = other.SlowFrames;
        FastFrames = other.FastFrames;
    }

    public SlowFastLLaVAOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        MaxFrames = 64;
    }

    /// <summary>Gets or sets the number of slow pathway frames (high-detail).</summary>
    public int SlowFrames { get; set; } = 8;

    /// <summary>Gets or sets the number of fast pathway frames (low-detail).</summary>
    public int FastFrames { get; set; } = 64;
}
