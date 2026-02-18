using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for DeepSeek-VL2 (MoE, dynamic tiling, multi-head latent attention).</summary>
public class DeepSeekVL2Options : InstructionTunedVLMOptions
{
    public DeepSeekVL2Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 60; NumHeads = 32; ImageSize = 384; LanguageModelName = "DeepSeek-MoE"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether dynamic tiling is enabled.</summary>
    public bool EnableDynamicTiling { get; set; } = true;
    /// <summary>Gets or sets the number of MoE experts.</summary>
    public int NumExperts { get; set; } = 64;
    /// <summary>Gets or sets the number of active MoE experts per token.</summary>
    public int NumActiveExperts { get; set; } = 6;
}
