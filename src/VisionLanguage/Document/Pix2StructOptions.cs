namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for Pix2Struct: screenshot parsing pre-training for visual language understanding.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Pix2Struct model. Default values follow the original paper settings.</para>
/// </remarks>
public class Pix2StructOptions : DocumentVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Pix2StructOptions(Pix2StructOptions other)
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
        MaxPatchesPerImage = other.MaxPatchesPerImage;
        EnableVariableResolution = other.EnableVariableResolution;
    }

    public Pix2StructOptions()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 1024;
    }

    /// <summary>Gets or sets the maximum number of image patches.</summary>
    public int MaxPatchesPerImage { get; set; } = 2048;

    /// <summary>Gets or sets whether to use variable-resolution patching.</summary>
    public bool EnableVariableResolution { get; set; } = true;
}
