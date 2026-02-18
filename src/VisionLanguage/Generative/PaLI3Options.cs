using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for PaLI-3 (efficient PaLI with SigLIP ViT, smaller and better).</summary>
public class PaLI3Options : GenerativeVLMOptions
{
    public PaLI3Options() { ArchitectureType = GenerativeArchitectureType.EncoderDecoder; VisionDim = 1152; DecoderDim = 1024; NumVisionLayers = 27; NumDecoderLayers = 24; NumHeads = 16; ImageSize = 224; }
    /// <summary>Gets or sets the number of decoder attention heads.</summary>
    public int NumDecoderHeads { get; set; } = 16;
}
