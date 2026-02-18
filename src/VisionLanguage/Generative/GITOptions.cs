using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for GIT (Generative Image-to-Text) with simple ViT + text decoder.</summary>
public class GITOptions : GenerativeVLMOptions
{
    public GITOptions() { ArchitectureType = GenerativeArchitectureType.EncoderDecoder; VisionDim = 768; DecoderDim = 768; NumVisionLayers = 12; NumDecoderLayers = 6; NumHeads = 12; }
}
