using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Base configuration options for document understanding vision-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Document model. Default values follow the original paper settings.</para>
/// </remarks>
public class DocumentVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DocumentVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DocumentVLMOptions(DocumentVLMOptions other)
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
    }

    /// <summary>Gets or sets whether this model operates OCR-free (no external OCR required).</summary>
    public bool IsOcrFree { get; set; } = true;

    /// <summary>Gets or sets the maximum document page count supported.</summary>
    public int MaxPages { get; set; } = 1;

    /// <summary>Gets or sets the maximum output text length in tokens.</summary>
    public int MaxOutputTokens { get; set; } = 4096;
}
