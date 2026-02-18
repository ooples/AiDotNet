using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Phi-4-Multimodal (unified vision + audio + text in single framework).</summary>
public class Phi4MultimodalOptions : InstructionTunedVLMOptions
{
    public Phi4MultimodalOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3072; ProjectionDim = 3072; NumVisionLayers = 27; NumDecoderLayers = 40; NumHeads = 32; ImageSize = 384; LanguageModelName = "Phi-4"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether audio input is enabled alongside vision.</summary>
    public bool EnableAudio { get; set; }
}
