using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for mPLUG-Owl (modular VLM with visual abstractor).</summary>
public class MPLUGOwlOptions : InstructionTunedVLMOptions
{
    public MPLUGOwlOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.VisualAbstractor; VisionDim = 1024; DecoderDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 224; LanguageModelName = "LLaMA"; MaxVisualTokens = 65; }
    /// <summary>Gets or sets the visual abstractor dimension.</summary>
    public int AbstractorDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of visual abstractor layers.</summary>
    public int NumAbstractorLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of visual abstractor attention heads.</summary>
    public int NumAbstractorHeads { get; set; } = 16;
}
