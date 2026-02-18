using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for CogVLM2 (improved architecture with video understanding, GLM-4/LLaMA-3).</summary>
public class CogVLM2Options : InstructionTunedVLMOptions
{
    public CogVLM2Options() { InstructionArchitectureType = InstructionTunedArchitectureType.VisualExpert; VisionDim = 1792; DecoderDim = 4096; NumVisionLayers = 63; NumDecoderLayers = 40; NumHeads = 32; ImageSize = 490; LanguageModelName = "GLM-4"; }
    /// <summary>Gets or sets the visual expert hidden dimension.</summary>
    public int VisualExpertDim { get; set; } = 4096;
    /// <summary>Gets or sets the number of visual expert attention heads.</summary>
    public int NumVisualExpertHeads { get; set; } = 32;
    /// <summary>Gets or sets whether video understanding is enabled.</summary>
    public bool EnableVideo { get; set; } = true;
}
