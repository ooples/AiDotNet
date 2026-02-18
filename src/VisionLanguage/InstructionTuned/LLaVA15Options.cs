using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for LLaVA-1.5 (MLP cross-modal connector, higher-res CLIP, academic task SOTA).</summary>
public class LLaVA15Options : InstructionTunedVLMOptions
{
    public LLaVA15Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "Vicuna"; MaxVisualTokens = 576; }
}
