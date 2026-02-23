using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// Base configuration options for 3D vision-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ThreeD model. Default values follow the original paper settings.</para>
/// </remarks>
public class ThreeDVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ThreeDVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ThreeDVLMOptions(ThreeDVLMOptions other)
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
    }

    /// <summary>Gets or sets the maximum number of 3D points the model can process.</summary>
    public int MaxPoints { get; set; } = 8192;

    /// <summary>Gets or sets the number of channels per point (3=XYZ, 6=XYZ+RGB, 9=XYZ+RGB+normals).</summary>
    public int PointChannels { get; set; } = 6;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the point cloud encoder hidden dimension.</summary>
    public int PointEncoderDim { get; set; } = 512;
}
