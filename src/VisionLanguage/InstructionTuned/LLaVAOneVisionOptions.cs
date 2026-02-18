using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for LLaVA-OneVision (single model for images, multi-image, and videos).</summary>
public class LLaVAOneVisionOptions : InstructionTunedVLMOptions
{
    public LLaVAOneVisionOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 27; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 384; LanguageModelName = "Qwen2"; MaxVisualTokens = 729; }
    /// <summary>Gets or sets whether video understanding is enabled.</summary>
    public bool EnableVideo { get; set; } = true;
    /// <summary>Gets or sets the maximum number of video frames.</summary>
    public int MaxVideoFrames { get; set; } = 32;
}
