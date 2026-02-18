using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Molmo (AI2 open VLM with diverse data mixture and pointing capability).</summary>
public class MolmoOptions : InstructionTunedVLMOptions
{
    public MolmoOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "OLMo"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether pointing capability (coordinate output) is enabled.</summary>
    public bool EnablePointing { get; set; } = true;
}
