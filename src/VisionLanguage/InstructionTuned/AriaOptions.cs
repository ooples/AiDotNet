using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Aria (Rhymes AI MoE VLM with 3.9B active params and 64K context multimodal).</summary>
public class AriaOptions : InstructionTunedVLMOptions
{
    public AriaOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 448; LanguageModelName = "Aria-MoE"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets the total number of MoE experts.</summary>
    public int NumExperts { get; set; } = 64;
    /// <summary>Gets or sets the number of active experts per token.</summary>
    public int NumActiveExperts { get; set; } = 8;
}
