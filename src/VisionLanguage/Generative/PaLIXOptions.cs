using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for PaLI-X (scaled PaLI to 55B with improved fine-tuning).</summary>
public class PaLIXOptions : GenerativeVLMOptions
{
    public PaLIXOptions() { ArchitectureType = GenerativeArchitectureType.EncoderDecoder; VisionDim = 4096; DecoderDim = 4096; NumVisionLayers = 48; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 224; }
}
