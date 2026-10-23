namespace AiDotNet.Audio.Features;

/// <summary>
/// Window types for spectral analysis.
/// </summary>
public enum WindowType
{
    /// <summary>Rectangular window (no tapering).</summary>
    Rectangular,

    /// <summary>Hann window (cosine tapering).</summary>
    Hann,

    /// <summary>Hamming window (raised cosine).</summary>
    Hamming,

    /// <summary>Blackman window (higher sidelobe attenuation).</summary>
    Blackman,

    /// <summary>Kaldi's Povey window: a Hann window raised to the power 0.85 (Kaldi's and torchaudio's fbank default).</summary>
    Povey
}
