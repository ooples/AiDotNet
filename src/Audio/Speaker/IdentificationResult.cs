namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Result of speaker identification.
/// </summary>
public class IdentificationResult
{
    /// <summary>
    /// Gets or sets the identified speaker ID (null if no match above threshold).
    /// </summary>
    public string? IdentifiedSpeakerId { get; set; }

    /// <summary>
    /// Gets or sets the top match score.
    /// </summary>
    public double TopScore { get; set; }

    /// <summary>
    /// Gets or sets the threshold used for identification.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Gets or sets ranked matches for all enrolled speakers.
    /// </summary>
    public List<SpeakerMatch> Matches { get; set; } = [];
}
