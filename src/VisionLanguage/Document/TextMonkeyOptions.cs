namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for TextMonkey: OCR-free text understanding with shifted window attention.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the TextMonkey model. Default values follow the original paper settings.</para>
/// </remarks>
public class TextMonkeyOptions : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TextMonkeyOptions(TextMonkeyOptions other)
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
        EnableShiftedWindowAttention = other.EnableShiftedWindowAttention;
    }

    public TextMonkeyOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 896;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use shifted window attention for text-heavy images.</summary>
    public bool EnableShiftedWindowAttention { get; set; } = true;
}
