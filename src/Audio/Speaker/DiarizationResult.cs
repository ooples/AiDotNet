namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Result of speaker diarization.
/// </summary>
public class DiarizationResult
{
    /// <summary>
    /// Gets or sets the speaker turns.
    /// </summary>
    public List<SpeakerTurn> Turns { get; set; } = [];

    /// <summary>
    /// Gets or sets the number of detected speakers.
    /// </summary>
    public int NumSpeakers { get; set; }

    /// <summary>
    /// Gets or sets the total audio duration in seconds.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets speaking time per speaker.
    /// </summary>
    public Dictionary<string, double> SpeakingTimePerSpeaker =>
        Turns.GroupBy(t => t.SpeakerId)
            .ToDictionary(g => g.Key, g => g.Sum(t => t.Duration));
}
