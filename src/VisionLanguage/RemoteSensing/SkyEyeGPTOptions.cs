using AiDotNet.VisionLanguage.RemoteSensing;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// Configuration options for SkyEyeGPT.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SkyEyeGPT model. Default values follow the original paper settings.</para>
/// </remarks>
public class SkyEyeGPTOptions : RemoteSensingVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SkyEyeGPTOptions(SkyEyeGPTOptions other)
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
        InstructionSamples = other.InstructionSamples;
    }

    public SkyEyeGPTOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        SupportedBands = "RGB";
    }

    /// <summary>Gets or sets the number of instruction-tuning samples.</summary>
    public int InstructionSamples { get; set; } = 968000;
}
