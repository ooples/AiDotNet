using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for LLaVA-NeXT (dynamic high-resolution via AnyRes, stronger reasoning).</summary>
public class LLaVANeXTOptions : InstructionTunedVLMOptions
{
    public LLaVANeXTOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 672; LanguageModelName = "LLaMA-3"; MaxVisualTokens = 2880; }
    /// <summary>Gets or sets whether AnyRes dynamic resolution is enabled.</summary>
    public bool EnableAnyRes { get; set; } = true;
    /// <summary>Gets or sets the maximum number of image tiles for AnyRes.</summary>
    public int MaxImageTiles { get; set; } = 5;
}
