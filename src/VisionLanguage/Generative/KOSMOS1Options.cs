using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for KOSMOS-1 (web-scale multimodal corpus with few-shot capabilities).</summary>
public class KOSMOS1Options : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KOSMOS1Options(KOSMOS1Options other)
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
    }

    public KOSMOS1Options() { ArchitectureType = GenerativeArchitectureType.CausalMultimodal; VisionDim = 1024; DecoderDim = 2048; NumVisionLayers = 24; NumDecoderLayers = 24; NumHeads = 32; }
}
