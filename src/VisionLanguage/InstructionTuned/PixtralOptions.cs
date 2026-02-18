using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Pixtral (12B decoder + 400M vision encoder from Mistral).</summary>
public class PixtralOptions : InstructionTunedVLMOptions
{
    public PixtralOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 1024; LanguageModelName = "Mistral"; MaxVisualTokens = 1024; }
}
