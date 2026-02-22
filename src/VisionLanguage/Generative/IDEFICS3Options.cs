using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for IDEFICS3 (state-of-the-art 8B trained on open datasets + Docmatix).</summary>
public class IDEFICS3Options : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public IDEFICS3Options(IDEFICS3Options other)
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
        PerceiverDim = other.PerceiverDim;
        NumPerceiverLayers = other.NumPerceiverLayers;
        NumLatents = other.NumLatents;
        NumPerceiverHeads = other.NumPerceiverHeads;
    }

    public IDEFICS3Options() { ArchitectureType = GenerativeArchitectureType.PerceiverResampler; VisionDim = 1152; DecoderDim = 4096; NumVisionLayers = 27; NumDecoderLayers = 32; NumHeads = 16; }
    /// <summary>Gets or sets the perceiver resampler dimension.</summary>
    public int PerceiverDim { get; set; } = 1152;
    /// <summary>Gets or sets the number of perceiver resampler layers.</summary>
    public int NumPerceiverLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of perceiver latent query tokens.</summary>
    public int NumLatents { get; set; } = 64;
    /// <summary>Gets or sets the number of perceiver attention heads.</summary>
    public int NumPerceiverHeads { get; set; } = 16;
}
