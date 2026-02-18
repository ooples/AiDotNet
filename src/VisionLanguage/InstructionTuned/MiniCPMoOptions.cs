using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for MiniCPM-o (omni-modal with real-time speech + vision).</summary>
public class MiniCPMoOptions : InstructionTunedVLMOptions
{
    public MiniCPMoOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 2304; ProjectionDim = 2304; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 36; ImageSize = 448; LanguageModelName = "MiniCPM"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether speech input is enabled.</summary>
    public bool EnableSpeech { get; set; }
    /// <summary>Gets or sets whether real-time processing mode is enabled.</summary>
    public bool EnableRealtime { get; set; }
}
