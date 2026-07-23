using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Configuration options for 3D-LLM.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ThreeDLLM model. Default values follow the original paper settings.</para>
/// </remarks>
public class ThreeDLLMOptions : ThreeDVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ThreeDLLMOptions(ThreeDLLMOptions other)
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
        NumViews = other.NumViews;
    }

    public ThreeDLLMOptions()
    {
        VisionDim = 768;
        DecoderDim = 4096;
        NumVisionLayers = 12;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        MaxPoints = 16384;
        PointChannels = 6;
        PointEncoderDim = 768;
    }

    /// <summary>Gets or sets the number of multi-view images for 3D feature extraction.</summary>
    public int NumViews { get; set; } = 8;
}
