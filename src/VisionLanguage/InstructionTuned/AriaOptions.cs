using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Aria (Rhymes AI MoE VLM with 3.9B active params and 64K context multimodal).</summary>
public class AriaOptions : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AriaOptions(AriaOptions other)
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
        NumExperts = other.NumExperts;
        NumActiveExperts = other.NumActiveExperts;
    }

    public AriaOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 448; LanguageModelName = "Aria-MoE"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets the total number of MoE experts.</summary>
    public int NumExperts { get; set; } = 64;
    /// <summary>Gets or sets the number of active experts per token.</summary>
    public int NumActiveExperts { get; set; } = 8;
}
