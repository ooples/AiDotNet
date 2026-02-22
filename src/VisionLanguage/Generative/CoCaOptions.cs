using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for CoCa (Contrastive Captioners) with dual loss.</summary>
public class CoCaOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CoCaOptions(CoCaOptions other)
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
        ContrastiveWeight = other.ContrastiveWeight;
        CaptionWeight = other.CaptionWeight;
    }

    public CoCaOptions() { ArchitectureType = GenerativeArchitectureType.EncoderDecoder; VisionDim = 1024; DecoderDim = 1024; NumVisionLayers = 24; NumDecoderLayers = 12; NumHeads = 16; }
    /// <summary>Gets or sets the contrastive loss weight.</summary>
    public double ContrastiveWeight { get; set; } = 1.0;
    /// <summary>Gets or sets the captioning loss weight.</summary>
    public double CaptionWeight { get; set; } = 1.0;
}
