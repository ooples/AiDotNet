using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Mantis (multi-image reasoning with interleaved data).</summary>
public class MantisOptions : InstructionTunedVLMOptions
{
    public MantisOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "LLaMA-3"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets the maximum number of images for multi-image reasoning.</summary>
    public int MaxImages { get; set; } = 16;
}
