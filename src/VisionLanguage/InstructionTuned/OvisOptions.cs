using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Ovis (structural embedding alignment for visual tokens).</summary>
public class OvisOptions : InstructionTunedVLMOptions
{
    public OvisOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 27; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 384; LanguageModelName = "Qwen2.5"; MaxVisualTokens = 576; }
}
