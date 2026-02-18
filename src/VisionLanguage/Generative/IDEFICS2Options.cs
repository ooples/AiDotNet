using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for IDEFICS2 (8B; SigLIP + Mistral-7B with Perceiver pooling).</summary>
public class IDEFICS2Options : GenerativeVLMOptions
{
    public IDEFICS2Options() { ArchitectureType = GenerativeArchitectureType.PerceiverResampler; VisionDim = 1152; DecoderDim = 4096; NumVisionLayers = 27; NumDecoderLayers = 32; NumHeads = 16; }
    /// <summary>Gets or sets the perceiver resampler dimension.</summary>
    public int PerceiverDim { get; set; } = 1152;
    /// <summary>Gets or sets the number of perceiver resampler layers.</summary>
    public int NumPerceiverLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of perceiver latent query tokens.</summary>
    public int NumLatents { get; set; } = 64;
    /// <summary>Gets or sets the number of perceiver attention heads.</summary>
    public int NumPerceiverHeads { get; set; } = 16;
}
