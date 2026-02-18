using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for KOSMOS-1 (web-scale multimodal corpus with few-shot capabilities).</summary>
public class KOSMOS1Options : GenerativeVLMOptions
{
    public KOSMOS1Options() { ArchitectureType = GenerativeArchitectureType.CausalMultimodal; VisionDim = 1024; DecoderDim = 2048; NumVisionLayers = 24; NumDecoderLayers = 24; NumHeads = 32; }
}
