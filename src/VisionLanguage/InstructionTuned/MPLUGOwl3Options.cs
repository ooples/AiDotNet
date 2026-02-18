using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for mPLUG-Owl3 (enhanced with hyper-attention for long visual sequences).</summary>
public class MPLUGOwl3Options : InstructionTunedVLMOptions
{
    public MPLUGOwl3Options() { InstructionArchitectureType = InstructionTunedArchitectureType.VisualAbstractor; VisionDim = 1024; DecoderDim = 3584; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 448; LanguageModelName = "Qwen2"; MaxVisualTokens = 128; }
    /// <summary>Gets or sets the visual abstractor dimension.</summary>
    public int AbstractorDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of visual abstractor layers.</summary>
    public int NumAbstractorLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of visual abstractor attention heads.</summary>
    public int NumAbstractorHeads { get; set; } = 16;
    /// <summary>Gets or sets whether hyper-attention for long visual sequences is enabled.</summary>
    public bool EnableHyperAttention { get; set; } = true;
}
