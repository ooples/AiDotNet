using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for Emu2 (scaled to 37B; enhanced understanding + generation).</summary>
public class Emu2Options : GenerativeVLMOptions
{
    public Emu2Options() { ArchitectureType = GenerativeArchitectureType.UnifiedGeneration; VisionDim = 1408; DecoderDim = 5120; NumVisionLayers = 39; NumDecoderLayers = 60; NumHeads = 40; }
    /// <summary>Gets or sets the visual regression head dimension.</summary>
    public int RegressionDim { get; set; } = 1408;
    /// <summary>Gets or sets the number of regression head layers.</summary>
    public int NumRegressionLayers { get; set; } = 2;
}
