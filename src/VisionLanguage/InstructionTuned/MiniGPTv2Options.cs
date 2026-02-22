using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for MiniGPT-v2 (multi-task with task-specific tokens, LLaMA-2 backbone).</summary>
public class MiniGPTv2Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MiniGPTv2Options(MiniGPTv2Options other)
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
        QFormerDim = other.QFormerDim;
        NumQFormerLayers = other.NumQFormerLayers;
        NumQueryTokens = other.NumQueryTokens;
        NumQFormerHeads = other.NumQFormerHeads;
    }

    public MiniGPTv2Options() { InstructionArchitectureType = InstructionTunedArchitectureType.QFormerProjection; VisionDim = 1408; DecoderDim = 4096; NumVisionLayers = 39; NumDecoderLayers = 32; NumHeads = 32; LanguageModelName = "LLaMA-2"; }
    /// <summary>Gets or sets the Q-Former dimension.</summary>
    public int QFormerDim { get; set; } = 768;
    /// <summary>Gets or sets the number of Q-Former layers.</summary>
    public int NumQFormerLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of learnable query tokens.</summary>
    public int NumQueryTokens { get; set; } = 32;
    /// <summary>Gets or sets the number of Q-Former attention heads.</summary>
    public int NumQFormerHeads { get; set; } = 12;
}
