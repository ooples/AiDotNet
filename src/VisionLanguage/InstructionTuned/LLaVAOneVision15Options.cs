using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for LLaVA-OneVision-1.5 (fully open, outperforms Qwen2.5-VL on 18/27 benchmarks).</summary>
public class LLaVAOneVision15Options : InstructionTunedVLMOptions
{
    public LLaVAOneVision15Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 27; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 384; LanguageModelName = "Qwen2.5"; MaxVisualTokens = 729; }
    /// <summary>Gets or sets whether video understanding is enabled.</summary>
    public bool EnableVideo { get; set; } = true;
    /// <summary>Gets or sets the maximum number of video frames.</summary>
    public int MaxVideoFrames { get; set; } = 64;
}
