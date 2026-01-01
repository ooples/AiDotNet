namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for speaker verification.
/// </summary>
public class SpeakerVerifierOptions
{
    /// <summary>
    /// Gets or sets the threshold for verification (0-1).
    /// </summary>
    public double VerificationThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for identification (0-1).
    /// </summary>
    public double IdentificationThreshold { get; set; } = 0.6;
}
