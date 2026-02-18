using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Llama 3.2 Vision (11B/90B vision models for edge/mobile deployment).</summary>
public class Llama32VisionOptions : InstructionTunedVLMOptions
{
    public Llama32VisionOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "LLaMA-3.2"; MaxVisualTokens = 576; }
}
