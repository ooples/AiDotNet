using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for KOSMOS-2 (grounded multimodal with text spans linked to bounding boxes).</summary>
public class KOSMOS2Options : GenerativeVLMOptions
{
    public KOSMOS2Options() { ArchitectureType = GenerativeArchitectureType.CausalMultimodal; VisionDim = 1024; DecoderDim = 2048; NumVisionLayers = 24; NumDecoderLayers = 24; NumHeads = 32; }
    /// <summary>Gets or sets whether location tokens for grounding are enabled.</summary>
    public bool EnableGroundingTokens { get; set; } = true;
    /// <summary>Gets or sets the number of location token bins for bounding box coordinates.</summary>
    public int NumLocationBins { get; set; } = 1000;
}
