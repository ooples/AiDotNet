using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for VILA (NVIDIA visual language models with joint pre-training on interleaved data).</summary>
public class VILAOptions : InstructionTunedVLMOptions
{
    public VILAOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 384; LanguageModelName = "LLaMA-3"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether interleaved image-text data training is enabled.</summary>
    public bool EnableInterleavedData { get; set; } = true;
}
