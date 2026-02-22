using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for PaLI-3 (efficient PaLI with SigLIP ViT, smaller and better).</summary>
public class PaLI3Options : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PaLI3Options(PaLI3Options other)
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

    public PaLI3Options() { ArchitectureType = GenerativeArchitectureType.EncoderDecoder; VisionDim = 1152; DecoderDim = 1024; NumVisionLayers = 27; NumDecoderLayers = 24; NumHeads = 16; ImageSize = 224; }
}
