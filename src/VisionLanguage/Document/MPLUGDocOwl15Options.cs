namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for mPLUG-DocOwl 1.5: unified structure learning achieving SOTA on 10 document benchmarks.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MPLUGDocOwl15 model. Default values follow the original paper settings.</para>
/// </remarks>
public class MPLUGDocOwl15Options : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MPLUGDocOwl15Options(MPLUGDocOwl15Options other)
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
        EnableUnifiedStructureLearning = other.EnableUnifiedStructureLearning;
    }

    public MPLUGDocOwl15Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
    }

    public int AbstractorDim { get; set; } = 1024;

    public int NumAbstractorLayers { get; set; } = 6;

    /// <summary>Gets or sets whether to use unified structure learning.</summary>
    public bool EnableUnifiedStructureLearning { get; set; } = true;
}
