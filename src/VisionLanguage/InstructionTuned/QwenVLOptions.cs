using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Qwen-VL (visual window attention, multi-resolution, bounding box output).</summary>
public class QwenVLOptions : InstructionTunedVLMOptions
{
    public QwenVLOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.CrossAttentionResampler; VisionDim = 1024; DecoderDim = 4096; NumVisionLayers = 48; NumDecoderLayers = 32; NumHeads = 16; ImageSize = 448; LanguageModelName = "Qwen"; MaxVisualTokens = 256; }
    /// <summary>Gets or sets the resampler dimension.</summary>
    public int ResamplerDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of resampler layers.</summary>
    public int NumResamplerLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of resampler heads.</summary>
    public int NumResamplerHeads { get; set; } = 16;
}
