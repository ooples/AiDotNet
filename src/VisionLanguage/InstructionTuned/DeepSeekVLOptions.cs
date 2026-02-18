using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for DeepSeek-VL (hybrid vision encoder with SigLIP + SAM-B).</summary>
public class DeepSeekVLOptions : InstructionTunedVLMOptions
{
    public DeepSeekVLOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 30; NumHeads = 32; ImageSize = 384; LanguageModelName = "DeepSeek"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether the hybrid encoder (SigLIP + SAM-B) is used.</summary>
    public bool UseHybridEncoder { get; set; } = true;
}
