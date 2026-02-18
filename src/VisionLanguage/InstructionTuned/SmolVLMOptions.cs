using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for SmolVLM (tiny efficient VLMs: 256M/500M/2.2B from HuggingFace).</summary>
public class SmolVLMOptions : InstructionTunedVLMOptions
{
    public SmolVLMOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 384; DecoderDim = 576; ProjectionDim = 576; NumVisionLayers = 12; NumDecoderLayers = 16; NumHeads = 9; ImageSize = 384; LanguageModelName = "SmolLM"; MaxVisualTokens = 256; }
    /// <summary>Gets or sets the model size variant (e.g., "256M", "500M", "2.2B").</summary>
    public string ModelVariant { get; set; } = "2.2B";
}
