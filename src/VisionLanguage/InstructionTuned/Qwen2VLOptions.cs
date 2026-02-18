using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Qwen2-VL (Naive Dynamic Resolution, M-RoPE for multimodal position).</summary>
public class Qwen2VLOptions : InstructionTunedVLMOptions
{
    public Qwen2VLOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.CrossAttentionResampler; VisionDim = 1152; DecoderDim = 3584; NumVisionLayers = 32; NumDecoderLayers = 28; NumHeads = 16; ImageSize = 448; LanguageModelName = "Qwen2"; MaxVisualTokens = 1024; }
    /// <summary>Gets or sets the resampler dimension.</summary>
    public int ResamplerDim { get; set; } = 1152;
    /// <summary>Gets or sets the number of resampler layers.</summary>
    public int NumResamplerLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of resampler heads.</summary>
    public int NumResamplerHeads { get; set; } = 16;
    /// <summary>Gets or sets whether M-RoPE (Multimodal RoPE) is enabled.</summary>
    public bool EnableMRoPE { get; set; } = true;
}
