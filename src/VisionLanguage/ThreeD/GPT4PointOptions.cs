using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for GPT4Point.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GPT4Point model. Default values follow the original paper settings.</para>
/// </remarks>
public class GPT4PointOptions : ThreeDVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GPT4PointOptions(GPT4PointOptions other)
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
        MaxPoints = other.MaxPoints;
        PointChannels = other.PointChannels;
        LanguageModelName = other.LanguageModelName;
        PointEncoderDim = other.PointEncoderDim;
        SupportsPointGeneration = other.SupportsPointGeneration;
    }

    public GPT4PointOptions()
    {
        VisionDim = 512;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        MaxPoints = 8192;
        PointChannels = 6;
        PointEncoderDim = 512;
    }

    /// <summary>Gets or sets whether the model supports point cloud generation.</summary>
    public bool SupportsPointGeneration { get; set; } = true;
}
