namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for Nougat: neural OCR for academic documents converting PDF to Markdown.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Nougat model. Default values follow the original paper settings.</para>
/// </remarks>
public class NougatOptions : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public NougatOptions(NougatOptions other)
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
        OutputFormat = other.OutputFormat;
    }

    public NougatOptions()
    {
        VisionDim = 1024;
        DecoderDim = 1024;
        NumVisionLayers = 12;
        NumDecoderLayers = 4;
        NumHeads = 16;
        ImageSize = 896;
    }

    /// <summary>Gets or sets the output format (Markdown, LaTeX, etc.).</summary>
    public string OutputFormat { get; set; } = "Markdown";
}
