using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Gemma 3 (3B-72B, Native Dynamic Resolution ViT, 128k context, 29 languages).</summary>
public class Gemma3Options : InstructionTunedVLMOptions
{
    public Gemma3Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 27; NumDecoderLayers = 36; NumHeads = 16; ImageSize = 896; LanguageModelName = "Gemma-3"; MaxVisualTokens = 4096; }
    /// <summary>Gets or sets whether native dynamic resolution is enabled.</summary>
    public bool EnableNativeDynamicResolution { get; set; } = true;
}
