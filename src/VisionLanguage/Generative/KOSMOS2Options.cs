using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for KOSMOS-2 (grounded multimodal with text spans linked to bounding boxes).</summary>
public class KOSMOS2Options : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KOSMOS2Options(KOSMOS2Options other)
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
        EnableGroundingTokens = other.EnableGroundingTokens;
        NumLocationBins = other.NumLocationBins;
    }

    public KOSMOS2Options() { ArchitectureType = GenerativeArchitectureType.CausalMultimodal; VisionDim = 1024; DecoderDim = 2048; NumVisionLayers = 24; NumDecoderLayers = 24; NumHeads = 32; }
    /// <summary>Gets or sets whether location tokens for grounding are enabled.</summary>
    public bool EnableGroundingTokens { get; set; } = true;
    /// <summary>Gets or sets the number of location token bins for bounding box coordinates.</summary>
    public int NumLocationBins { get; set; } = 1000;
}
