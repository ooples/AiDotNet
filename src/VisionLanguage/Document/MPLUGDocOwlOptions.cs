namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for mPLUG-DocOwl: modular MLLM for document understanding with visual abstractor.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MPLUGDocOwl model. Default values follow the original paper settings.</para>
/// </remarks>
public class MPLUGDocOwlOptions : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MPLUGDocOwlOptions(MPLUGDocOwlOptions other)
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
        IsOcrFree = other.IsOcrFree;
        MaxPages = other.MaxPages;
        MaxOutputTokens = other.MaxOutputTokens;
        AbstractorDim = other.AbstractorDim;
        NumAbstractorLayers = other.NumAbstractorLayers;
    }

    public MPLUGDocOwlOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets the visual abstractor dimension.</summary>
    public int AbstractorDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of abstractor layers.</summary>
    public int NumAbstractorLayers { get; set; } = 6;
}
