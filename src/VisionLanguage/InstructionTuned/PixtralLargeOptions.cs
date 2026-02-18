using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Pixtral Large (124B decoder + 1B vision encoder from Mistral).</summary>
public class PixtralLargeOptions : InstructionTunedVLMOptions
{
    public PixtralLargeOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1280; DecoderDim = 8192; ProjectionDim = 8192; NumVisionLayers = 32; NumDecoderLayers = 80; NumHeads = 64; ImageSize = 1024; LanguageModelName = "Mistral-Large"; MaxVisualTokens = 1024; }
}
