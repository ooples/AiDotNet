namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Result of beat tracking.
/// </summary>
public class BeatTrackingResult
{
    /// <summary>
    /// Gets or sets the estimated tempo in beats per minute.
    /// </summary>
    public double Tempo { get; set; }

    /// <summary>
    /// Gets or sets the times of detected beats in seconds.
    /// </summary>
    public List<double> BeatTimes { get; set; } = [];

    /// <summary>
    /// Gets or sets the confidence score (0-1).
    /// </summary>
    public double ConfidenceScore { get; set; }

    /// <summary>
    /// Gets the average beat interval in seconds.
    /// </summary>
    public double AverageBeatInterval => 60.0 / Tempo;
}
