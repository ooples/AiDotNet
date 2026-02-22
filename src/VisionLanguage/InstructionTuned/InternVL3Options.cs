using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for InternVL3 (78B, MMMU 72.2 SOTA among open-source).</summary>
public class InternVL3Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public InternVL3Options(InternVL3Options other)
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
        InstructionArchitectureType = other.InstructionArchitectureType;
        ProjectionDim = other.ProjectionDim;
        LanguageModelName = other.LanguageModelName;
        MaxVisualTokens = other.MaxVisualTokens;
        SystemPrompt = other.SystemPrompt;
        PixelShuffleFactor = other.PixelShuffleFactor;
        EnableDynamicResolution = other.EnableDynamicResolution;
    }

    public InternVL3Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 3200; DecoderDim = 8192; ProjectionDim = 8192; NumVisionLayers = 48; NumDecoderLayers = 80; NumHeads = 25; ImageSize = 448; LanguageModelName = "InternLM3"; MaxVisualTokens = 256; }
    /// <summary>Gets or sets the pixel shuffle downscale factor.</summary>
    public int PixelShuffleFactor { get; set; } = 2;
    /// <summary>Gets or sets whether dynamic resolution is enabled.</summary>
    public bool EnableDynamicResolution { get; set; } = true;
}
