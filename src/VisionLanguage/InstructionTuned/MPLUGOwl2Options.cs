using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for mPLUG-Owl2 (improved modular design for multi-image understanding).</summary>
public class MPLUGOwl2Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MPLUGOwl2Options(MPLUGOwl2Options other)
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
        AbstractorDim = other.AbstractorDim;
        NumAbstractorLayers = other.NumAbstractorLayers;
        NumAbstractorHeads = other.NumAbstractorHeads;
    }

    public MPLUGOwl2Options() { InstructionArchitectureType = InstructionTunedArchitectureType.VisualAbstractor; VisionDim = 1024; DecoderDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 448; LanguageModelName = "LLaMA-2"; MaxVisualTokens = 65; }
    /// <summary>Gets or sets the visual abstractor dimension.</summary>
    public int AbstractorDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of visual abstractor layers.</summary>
    public int NumAbstractorLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of visual abstractor attention heads.</summary>
    public int NumAbstractorHeads { get; set; } = 16;
}
