namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for GOT-OCR2: 580M unified OCR model for text, tables, charts, equations, and music scores.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GOTOCR2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class GOTOCR2Options : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GOTOCR2Options(GOTOCR2Options other)
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
        EnableMathOCR = other.EnableMathOCR;
        EnableMusicOCR = other.EnableMusicOCR;
    }

    public GOTOCR2Options()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 1024;
    }

    /// <summary>Gets or sets whether to enable mathematical equation OCR.</summary>
    public bool EnableMathOCR { get; set; } = true;

    /// <summary>Gets or sets whether to enable music score OCR.</summary>
    public bool EnableMusicOCR { get; set; } = true;
}
