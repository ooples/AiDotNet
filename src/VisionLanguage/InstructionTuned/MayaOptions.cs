using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Maya (multilingual VLM for low-resource languages).</summary>
public class MayaOptions : InstructionTunedVLMOptions
{
    public MayaOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "LLaMA-2"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets the number of supported languages.</summary>
    public int NumLanguages { get; set; } = 8;
}
