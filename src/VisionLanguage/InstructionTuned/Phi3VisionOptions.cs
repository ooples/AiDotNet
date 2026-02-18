using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Phi-3-Vision (compact 4.2B with strong vision via curated data).</summary>
public class Phi3VisionOptions : InstructionTunedVLMOptions
{
    public Phi3VisionOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 3072; ProjectionDim = 3072; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "Phi-3"; MaxVisualTokens = 576; }
}
