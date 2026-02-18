using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Moondream (1.8B; SigLIP + Phi-1.5; ideal for edge/mobile).</summary>
public class MoondreamOptions : InstructionTunedVLMOptions
{
    public MoondreamOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 2048; ProjectionDim = 2048; NumVisionLayers = 24; NumDecoderLayers = 24; NumHeads = 32; ImageSize = 378; LanguageModelName = "Phi-1.5"; MaxVisualTokens = 576; }
}
