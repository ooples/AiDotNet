using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for 3DGraphLLM.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ThreeDGraphLLM model. Default values follow the original paper settings.</para>
/// </remarks>
public class ThreeDGraphLLMOptions : ThreeDVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ThreeDGraphLLMOptions(ThreeDGraphLLMOptions other)
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
        MaxGraphNodes = other.MaxGraphNodes;
    }

    public ThreeDGraphLLMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        MaxPoints = 8192;
        PointChannels = 6;
        PointEncoderDim = 768;
    }

    /// <summary>Gets or sets the maximum number of graph nodes.</summary>
    public int MaxGraphNodes { get; set; } = 256;
}
