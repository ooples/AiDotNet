namespace AiDotNet.Audio.Features;

/// <summary>
/// Options for spectral feature extraction.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpectralFeature model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpectralFeatureOptions : AudioFeatureOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SpectralFeatureOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SpectralFeatureOptions(SpectralFeatureOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        FeatureTypes = other.FeatureTypes;
        RolloffPercentage = other.RolloffPercentage;
    }

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
