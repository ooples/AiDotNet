namespace AiDotNet.Audio.Speaker;

/// <summary>
/// A speaker match with score.
/// </summary>
public class SpeakerMatch
{
    /// <summary>
    /// Gets or sets the speaker ID.
    /// </summary>
    public string SpeakerId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the similarity score.
    /// </summary>
    public double Score { get; set; }
}
