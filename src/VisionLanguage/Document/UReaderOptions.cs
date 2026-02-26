namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for UReader: universal OCR-free visually-situated language model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UReader model. Default values follow the original paper settings.</para>
/// </remarks>
public class UReaderOptions : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UReaderOptions(UReaderOptions other)
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
        EnableShapeAdaptiveCropping = other.EnableShapeAdaptiveCropping;
    }

    public UReaderOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use shape-adaptive cropping for varied document layouts.</summary>
    public bool EnableShapeAdaptiveCropping { get; set; } = true;
}
