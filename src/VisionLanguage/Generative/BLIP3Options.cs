using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for BLIP-3 (xGen-MM) with interleaved data and any-to-any generation.</summary>
public class BLIP3Options : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BLIP3Options(BLIP3Options other)
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
        QFormerDim = other.QFormerDim;
        NumQFormerLayers = other.NumQFormerLayers;
        NumQueryTokens = other.NumQueryTokens;
        NumQFormerHeads = other.NumQFormerHeads;
        UseInterleavedData = other.UseInterleavedData;
    }

    public BLIP3Options() { ArchitectureType = GenerativeArchitectureType.QFormerBridge; VisionDim = 1408; DecoderDim = 4096; NumVisionLayers = 39; NumDecoderLayers = 32; NumHeads = 16; }
    /// <summary>Gets or sets the Q-Former hidden dimension.</summary>
    public int QFormerDim { get; set; } = 768;
    /// <summary>Gets or sets the number of Q-Former layers.</summary>
    public int NumQFormerLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of learnable query tokens.</summary>
    public int NumQueryTokens { get; set; } = 64;
    /// <summary>Gets or sets the number of Q-Former attention heads.</summary>
    public int NumQFormerHeads { get; set; } = 12;
    /// <summary>Gets or sets whether interleaved image-text training is enabled.</summary>
    public bool UseInterleavedData { get; set; } = true;
}
