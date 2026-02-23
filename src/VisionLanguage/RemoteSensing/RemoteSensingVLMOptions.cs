using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// Base configuration options for remote sensing vision-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the RemoteSensing model. Default values follow the original paper settings.</para>
/// </remarks>
public class RemoteSensingVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public RemoteSensingVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RemoteSensingVLMOptions(RemoteSensingVLMOptions other)
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
        SupportedBands = other.SupportedBands;
        LanguageModelName = other.LanguageModelName;
        GroundSampleDistance = other.GroundSampleDistance;
    }

    /// <summary>Gets or sets the supported image bands (e.g., "RGB", "Multispectral").</summary>
    public string SupportedBands { get; set; } = "RGB";

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the ground sample distance in meters.</summary>
    public double GroundSampleDistance { get; set; } = 0.5;
}
