namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for LayoutLMv3: unified text, image, and layout pre-training for document AI.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the LayoutLMv3 model. Default values follow the original paper settings.</para>
/// </remarks>
public class LayoutLMv3Options : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LayoutLMv3Options(LayoutLMv3Options other)
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
        MaxLayoutTokens = other.MaxLayoutTokens;
    }

    public LayoutLMv3Options()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 224;
        IsOcrFree = false;
    }

    /// <summary>Gets or sets the maximum number of layout tokens.</summary>
    public int MaxLayoutTokens { get; set; } = 512;
}
