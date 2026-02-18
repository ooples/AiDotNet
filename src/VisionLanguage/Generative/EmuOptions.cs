using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for Emu (unified VQA, captioning, generation via EVA-CLIP + LLM + diffusion).</summary>
public class EmuOptions : GenerativeVLMOptions
{
    public EmuOptions() { ArchitectureType = GenerativeArchitectureType.UnifiedGeneration; VisionDim = 1408; DecoderDim = 4096; NumVisionLayers = 39; NumDecoderLayers = 32; NumHeads = 32; }
    /// <summary>Gets or sets the visual regression head dimension (for image generation).</summary>
    public int RegressionDim { get; set; } = 1408;
    /// <summary>Gets or sets the number of regression head layers.</summary>
    public int NumRegressionLayers { get; set; } = 2;
}
