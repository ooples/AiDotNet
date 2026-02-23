using AiDotNet.Models.Options;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for speaker verification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpeakerVerifier model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeakerVerifierOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SpeakerVerifierOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SpeakerVerifierOptions(SpeakerVerifierOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        VerificationThreshold = other.VerificationThreshold;
        IdentificationThreshold = other.IdentificationThreshold;
    }

    /// <summary>
    /// Gets or sets the threshold for verification (0-1).
    /// </summary>
    public double VerificationThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for identification (0-1).
    /// </summary>
    public double IdentificationThreshold { get; set; } = 0.6;
}
