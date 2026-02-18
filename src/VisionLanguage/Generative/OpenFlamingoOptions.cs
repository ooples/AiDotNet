using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for OpenFlamingo (open-source Flamingo with perceiver resampler).</summary>
public class OpenFlamingoOptions : GenerativeVLMOptions
{
    public OpenFlamingoOptions() { ArchitectureType = GenerativeArchitectureType.PerceiverResampler; VisionDim = 1024; DecoderDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 16; }
    /// <summary>Gets or sets the perceiver resampler dimension.</summary>
    public int PerceiverDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of perceiver resampler layers.</summary>
    public int NumPerceiverLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of perceiver latent query tokens.</summary>
    public int NumLatents { get; set; } = 64;
    /// <summary>Gets or sets the number of perceiver attention heads.</summary>
    public int NumPerceiverHeads { get; set; } = 16;
}
