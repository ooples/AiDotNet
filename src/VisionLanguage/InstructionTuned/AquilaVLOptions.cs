using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Aquila-VL (BAAI LLaVA-style VLM with Qwen2.5 + SigLIP).</summary>
public class AquilaVLOptions : InstructionTunedVLMOptions
{
    public AquilaVLOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 27; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 448; LanguageModelName = "Qwen2.5"; MaxVisualTokens = 576; }
}
