using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for MiniGPT-4 (ViT + Q-Former aligned with Vicuna via single projection layer).</summary>
public class MiniGPT4Options : InstructionTunedVLMOptions
{
    public MiniGPT4Options() { InstructionArchitectureType = InstructionTunedArchitectureType.QFormerProjection; VisionDim = 1408; DecoderDim = 4096; NumVisionLayers = 39; NumDecoderLayers = 32; NumHeads = 32; LanguageModelName = "Vicuna"; }
    /// <summary>Gets or sets the Q-Former dimension.</summary>
    public int QFormerDim { get; set; } = 768;
    /// <summary>Gets or sets the number of Q-Former layers.</summary>
    public int NumQFormerLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of learnable query tokens.</summary>
    public int NumQueryTokens { get; set; } = 32;
    /// <summary>Gets or sets the number of Q-Former attention heads.</summary>
    public int NumQFormerHeads { get; set; } = 12;
}
