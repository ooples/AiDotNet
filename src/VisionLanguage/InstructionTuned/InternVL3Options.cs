using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for InternVL3 (78B, MMMU 72.2 SOTA among open-source).</summary>
public class InternVL3Options : InstructionTunedVLMOptions
{
    public InternVL3Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 3200; DecoderDim = 8192; ProjectionDim = 8192; NumVisionLayers = 48; NumDecoderLayers = 80; NumHeads = 25; ImageSize = 448; LanguageModelName = "InternLM3"; MaxVisualTokens = 256; }
    /// <summary>Gets or sets the pixel shuffle downscale factor.</summary>
    public int PixelShuffleFactor { get; set; } = 2;
    /// <summary>Gets or sets whether dynamic resolution is enabled.</summary>
    public bool EnableDynamicResolution { get; set; } = true;
}
