namespace AiDotNet.Audio.Features;

/// <summary>
/// Options for spectral feature extraction.
/// </summary>
public class SpectralFeatureOptions : AudioFeatureOptions
{
    /// <summary>
    /// Gets or sets which spectral features to extract.
    /// Default is Basic (centroid, bandwidth, rolloff, flux, flatness).
    /// </summary>
    public SpectralFeatureType FeatureTypes { get; set; } = SpectralFeatureType.Basic;

    /// <summary>
    /// Gets or sets the rolloff percentage (0-1).
    /// Default is 0.85 (85th percentile).
    /// </summary>
    public double RolloffPercentage { get; set; } = 0.85;
}
