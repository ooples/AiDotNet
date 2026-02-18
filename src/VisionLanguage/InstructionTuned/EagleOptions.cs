using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Eagle (NVIDIA data-centric VLM with learned data mixing).</summary>
public class EagleOptions : InstructionTunedVLMOptions
{
    public EagleOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 448; LanguageModelName = "LLaMA-3"; MaxVisualTokens = 576; }
}
