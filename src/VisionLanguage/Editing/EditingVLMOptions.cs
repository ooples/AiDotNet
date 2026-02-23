using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Base configuration options for image editing VLMs.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Editing model. Default values follow the original paper settings.</para>
/// </remarks>
public class EditingVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EditingVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EditingVLMOptions(EditingVLMOptions other)
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
        OutputImageSize = other.OutputImageSize;
        NumDiffusionSteps = other.NumDiffusionSteps;
        GuidanceScale = other.GuidanceScale;
    }

    /// <summary>Gets or sets the output image resolution.</summary>
    public int OutputImageSize { get; set; } = 512;

    /// <summary>Gets or sets the number of diffusion denoising steps.</summary>
    public int NumDiffusionSteps { get; set; } = 50;

    /// <summary>Gets or sets the guidance scale for classifier-free guidance.</summary>
    public double GuidanceScale { get; set; } = 7.5;
}
