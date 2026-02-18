using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for InstructBLIP (instruction-tuned Q-Former for zero-shot generalization).</summary>
public class InstructBLIPOptions : GenerativeVLMOptions
{
    public InstructBLIPOptions() { ArchitectureType = GenerativeArchitectureType.QFormerBridge; VisionDim = 1408; DecoderDim = 4096; NumVisionLayers = 39; NumDecoderLayers = 32; NumHeads = 16; }
    /// <summary>Gets or sets the Q-Former hidden dimension.</summary>
    public int QFormerDim { get; set; } = 768;
    /// <summary>Gets or sets the number of Q-Former layers.</summary>
    public int NumQFormerLayers { get; set; } = 12;
    /// <summary>Gets or sets the number of learnable query tokens.</summary>
    public int NumQueryTokens { get; set; } = 32;
    /// <summary>Gets or sets the number of Q-Former attention heads.</summary>
    public int NumQFormerHeads { get; set; } = 12;
}
