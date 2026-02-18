using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for CoCa (Contrastive Captioners) with dual loss.</summary>
public class CoCaOptions : GenerativeVLMOptions
{
    public CoCaOptions() { ArchitectureType = GenerativeArchitectureType.EncoderDecoder; VisionDim = 1024; DecoderDim = 1024; NumVisionLayers = 24; NumDecoderLayers = 12; NumHeads = 16; }
    /// <summary>Gets or sets the contrastive loss weight.</summary>
    public double ContrastiveWeight { get; set; } = 1.0;
    /// <summary>Gets or sets the captioning loss weight.</summary>
    public double CaptionWeight { get; set; } = 1.0;
}
