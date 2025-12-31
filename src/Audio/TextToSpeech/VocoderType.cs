namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Available vocoder types.
/// </summary>
public enum VocoderType
{
    /// <summary>HiFi-GAN vocoder (high quality, fast).</summary>
    HiFiGan,

    /// <summary>WaveGlow vocoder (high quality, slow).</summary>
    WaveGlow,

    /// <summary>Griffin-Lim (lower quality, no neural network).</summary>
    GriffinLim
}
