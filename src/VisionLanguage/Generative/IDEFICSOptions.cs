using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for IDEFICS (80B open reproduction of Flamingo).</summary>
public class IDEFICSOptions : GenerativeVLMOptions
{
    public IDEFICSOptions() { ArchitectureType = GenerativeArchitectureType.PerceiverResampler; VisionDim = 1024; DecoderDim = 5120; NumVisionLayers = 24; NumDecoderLayers = 60; NumHeads = 16; }
    /// <summary>Gets or sets the perceiver resampler dimension.</summary>
    public int PerceiverDim { get; set; } = 1024;
    /// <summary>Gets or sets the number of perceiver resampler layers.</summary>
    public int NumPerceiverLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of perceiver latent query tokens.</summary>
    public int NumLatents { get; set; } = 64;
    /// <summary>Gets or sets the number of perceiver attention heads.</summary>
    public int NumPerceiverHeads { get; set; } = 16;
}
