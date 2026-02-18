using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for MiniCPM-V (8B model achieving GPT-4V-level on select benchmarks).</summary>
public class MiniCPMVOptions : InstructionTunedVLMOptions
{
    public MiniCPMVOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 2304; ProjectionDim = 2304; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 36; ImageSize = 448; LanguageModelName = "MiniCPM"; MaxVisualTokens = 576; }
}
