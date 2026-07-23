using AiDotNet.VisionLanguage.RemoteSensing;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// Configuration options for RSGPT.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the RSGPT model. Default values follow the original paper settings.</para>
/// </remarks>
public class RSGPTOptions : RemoteSensingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RSGPTOptions(RSGPTOptions other)
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

    public RSGPTOptions()
    {
        VisionDim = 1408;
        DecoderDim = 4096;
        NumVisionLayers = 39;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "Vicuna";
        SupportedBands = "RGB";
    }
}
