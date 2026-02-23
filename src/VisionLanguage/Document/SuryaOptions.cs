namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for Surya: multi-language OCR with layout analysis support.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Surya model. Default values follow the original paper settings.</para>
/// </remarks>
public class SuryaOptions : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SuryaOptions(SuryaOptions other)
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
        NumLanguages = other.NumLanguages;
        EnableLayoutAnalysis = other.EnableLayoutAnalysis;
    }

    public SuryaOptions()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 6;
        NumHeads = 12;
        ImageSize = 896;
    }

    /// <summary>Gets or sets the number of supported languages.</summary>
    public int NumLanguages { get; set; } = 90;

    /// <summary>Gets or sets whether to perform layout analysis.</summary>
    public bool EnableLayoutAnalysis { get; set; } = true;
}
