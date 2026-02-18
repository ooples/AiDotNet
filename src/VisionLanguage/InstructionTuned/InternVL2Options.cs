using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for InternVL2 (dynamic resolution, pixel shuffle, InternLM2 backbone).</summary>
public class InternVL2Options : InstructionTunedVLMOptions
{
    public InternVL2Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 3200; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 48; NumDecoderLayers = 32; NumHeads = 25; ImageSize = 448; LanguageModelName = "InternLM2"; MaxVisualTokens = 256; }
    /// <summary>Gets or sets the pixel shuffle downscale factor.</summary>
    public int PixelShuffleFactor { get; set; } = 2;
    /// <summary>Gets or sets whether dynamic resolution is enabled.</summary>
    public bool EnableDynamicResolution { get; set; } = true;
}
