using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for CogVLM (deep fusion via visual expert module in every LLM layer).</summary>
public class CogVLMOptions : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CogVLMOptions(CogVLMOptions other)
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
        VisualExpertDim = other.VisualExpertDim;
        NumVisualExpertHeads = other.NumVisualExpertHeads;
    }

    public CogVLMOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.VisualExpert; VisionDim = 1792; DecoderDim = 4096; NumVisionLayers = 63; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 490; LanguageModelName = "Vicuna"; }
    /// <summary>Gets or sets the visual expert hidden dimension.</summary>
    public int VisualExpertDim { get; set; } = 4096;
    /// <summary>Gets or sets the number of visual expert attention heads.</summary>
    public int NumVisualExpertHeads { get; set; } = 32;
}
