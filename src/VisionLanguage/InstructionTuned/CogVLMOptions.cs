using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for CogVLM (deep fusion via visual expert module in every LLM layer).</summary>
public class CogVLMOptions : InstructionTunedVLMOptions
{
    public CogVLMOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.VisualExpert; VisionDim = 1792; DecoderDim = 4096; NumVisionLayers = 63; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 490; LanguageModelName = "Vicuna"; }
    /// <summary>Gets or sets the visual expert hidden dimension.</summary>
    public int VisualExpertDim { get; set; } = 4096;
    /// <summary>Gets or sets the number of visual expert attention heads.</summary>
    public int NumVisualExpertHeads { get; set; } = 32;
}
