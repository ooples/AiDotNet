using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for PaLI (Pathways Language and Image model) with ViT-e + mT5.</summary>
public class PaLIOptions : GenerativeVLMOptions
{
    public PaLIOptions() { ArchitectureType = GenerativeArchitectureType.EncoderDecoder; VisionDim = 1408; DecoderDim = 1024; NumVisionLayers = 48; NumDecoderLayers = 24; NumHeads = 16; ImageSize = 224; }
}
