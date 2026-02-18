using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Cambrian-1 (Spatial Vision Aggregator with 35+ vision encoder combinations).</summary>
public class Cambrian1Options : InstructionTunedVLMOptions
{
    public Cambrian1Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "LLaMA-3"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets the number of vision encoders used in the Spatial Vision Aggregator.</summary>
    public int NumVisionEncoders { get; set; } = 4;
    /// <summary>Gets or sets whether the Spatial Vision Aggregator is enabled.</summary>
    public bool EnableSpatialVisionAggregator { get; set; } = true;
}
