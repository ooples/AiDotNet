using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for Emu3 (next-token prediction unifies understanding + generation).</summary>
public class Emu3Options : GenerativeVLMOptions
{
    public Emu3Options() { ArchitectureType = GenerativeArchitectureType.UnifiedGeneration; VisionDim = 1408; DecoderDim = 4096; NumVisionLayers = 39; NumDecoderLayers = 32; NumHeads = 32; }
    /// <summary>Gets or sets the visual regression head dimension.</summary>
    public int RegressionDim { get; set; } = 1408;
    /// <summary>Gets or sets the number of regression head layers.</summary>
    public int NumRegressionLayers { get; set; } = 2;
}
