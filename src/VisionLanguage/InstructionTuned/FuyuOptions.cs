using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Fuyu (no vision encoder; raw patches directly into transformer).</summary>
public class FuyuOptions : InstructionTunedVLMOptions
{
    public FuyuOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.DirectPatch; VisionDim = 4096; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 0; NumDecoderLayers = 36; NumHeads = 64; ImageSize = 1080; LanguageModelName = "Fuyu"; MaxVisualTokens = 1296; }
    /// <summary>Gets or sets the patch size for raw image patches (default 30x30).</summary>
    public int PatchSize { get; set; } = 30;
}
