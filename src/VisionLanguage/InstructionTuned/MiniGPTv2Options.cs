using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for MiniGPT-v2 (multi-task with task-specific tokens, LLaMA-2 backbone).</summary>
public class MiniGPTv2Options : InstructionTunedVLMOptions
{
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
