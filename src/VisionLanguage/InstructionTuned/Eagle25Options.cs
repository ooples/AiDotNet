using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Eagle 2.5 (NVIDIA long-context multimodal for video + high-res images).</summary>
public class Eagle25Options : InstructionTunedVLMOptions
{
    public Eagle25Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 448; LanguageModelName = "Qwen2"; MaxVisualTokens = 2048; }
    /// <summary>Gets or sets the maximum number of video frames for long-context video understanding.</summary>
    public int MaxVideoFrames { get; set; } = 512;
    /// <summary>Gets or sets whether long-context mode is enabled.</summary>
    public bool EnableLongContext { get; set; } = true;
}
