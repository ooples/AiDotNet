using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for VILA-U (NVIDIA unified understanding + generation via shared visual embedding).</summary>
public class VILAUOptions : InstructionTunedVLMOptions
{
    public VILAUOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 384; LanguageModelName = "LLaMA-3"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether image generation capability is enabled.</summary>
    public bool EnableGeneration { get; set; } = true;
}
