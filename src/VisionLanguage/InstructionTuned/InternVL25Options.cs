using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for InternVL2.5 (improved training data and strategy over InternVL2).</summary>
public class InternVL25Options : InstructionTunedVLMOptions
{
    public InternVL25Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 3200; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 48; NumDecoderLayers = 32; NumHeads = 25; ImageSize = 448; LanguageModelName = "InternLM2.5"; MaxVisualTokens = 256; }
    /// <summary>Gets or sets the pixel shuffle downscale factor.</summary>
    public int PixelShuffleFactor { get; set; } = 2;
    /// <summary>Gets or sets whether dynamic resolution is enabled.</summary>
    public bool EnableDynamicResolution { get; set; } = true;
}
