using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for InternVL (6B InternViT + LLaMA with progressive alignment).</summary>
public class InternVLOptions : InstructionTunedVLMOptions
{
    public InternVLOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 3200; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 48; NumDecoderLayers = 32; NumHeads = 25; ImageSize = 448; LanguageModelName = "LLaMA"; MaxVisualTokens = 256; }
}
