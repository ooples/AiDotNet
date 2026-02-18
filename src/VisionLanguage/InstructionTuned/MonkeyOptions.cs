using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Monkey (multi-level description generation with high resolution).</summary>
public class MonkeyOptions : InstructionTunedVLMOptions
{
    public MonkeyOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 896; LanguageModelName = "Qwen"; MaxVisualTokens = 1024; }
    /// <summary>Gets or sets whether multi-level description generation is enabled.</summary>
    public bool EnableMultiLevelDescription { get; set; } = true;
}
