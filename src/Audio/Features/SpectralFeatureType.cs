namespace AiDotNet.Audio.Features;

/// <summary>
/// Types of spectral features that can be extracted.
/// </summary>
[Flags]
public enum SpectralFeatureType
{
    /// <summary>No features.</summary>
    None = 0,

    /// <summary>Spectral centroid (center of mass of spectrum).</summary>
    Centroid = 1,

    /// <summary>Spectral bandwidth (spread around centroid).</summary>
    Bandwidth = 2,

    /// <summary>Spectral rolloff (frequency below which most energy is concentrated).</summary>
    Rolloff = 4,

    /// <summary>Spectral flux (frame-to-frame spectral change).</summary>
    Flux = 8,

    /// <summary>Spectral flatness (noisiness vs tonality).</summary>
    Flatness = 16,

    /// <summary>Spectral contrast (difference between peaks and valleys in sub-bands).</summary>
    Contrast = 32,

    /// <summary>Zero crossing rate (sign change frequency).</summary>
    ZeroCrossingRate = 64,

    /// <summary>All basic features (centroid, bandwidth, rolloff, flux, flatness).</summary>
    Basic = Centroid | Bandwidth | Rolloff | Flux | Flatness,

    /// <summary>All available features.</summary>
    All = Centroid | Bandwidth | Rolloff | Flux | Flatness | Contrast | ZeroCrossingRate
}
