using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Qwen3-VL (latest series with 2B/4B/8B/32B variants).</summary>
public class Qwen3VLOptions : InstructionTunedVLMOptions
{
    public Qwen3VLOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.CrossAttentionResampler; VisionDim = 1152; DecoderDim = 3584; NumVisionLayers = 32; NumDecoderLayers = 28; NumHeads = 16; ImageSize = 448; LanguageModelName = "Qwen3"; MaxVisualTokens = 1024; }
    /// <summary>Gets or sets the resampler dimension.</summary>
    public int ResamplerDim { get; set; } = 1152;
    /// <summary>Gets or sets the number of resampler layers.</summary>
    public int NumResamplerLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of resampler heads.</summary>
    public int NumResamplerHeads { get; set; } = 16;
}
