using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for NVLM 1.0 (cross-attention + decoder-only hybrid that retains text performance).</summary>
public class NVLMOptions : InstructionTunedVLMOptions
{
    public NVLMOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 448; LanguageModelName = "Qwen2"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether cross-attention hybrid mode is enabled.</summary>
    public bool EnableCrossAttention { get; set; } = true;
    /// <summary>Gets or sets the cross-attention dimension.</summary>
    public int CrossAttentionDim { get; set; } = 3584;
}
