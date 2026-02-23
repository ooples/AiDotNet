namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for mPLUG-DocOwl 2: high-res compressing for multi-page document understanding.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MPLUGDocOwl2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class MPLUGDocOwl2Options : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MPLUGDocOwl2Options(MPLUGDocOwl2Options other)
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

    public MPLUGDocOwl2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        MaxPages = 20;
    }

    public int AbstractorDim { get; set; } = 1024;

    public int NumAbstractorLayers { get; set; } = 6;
}
