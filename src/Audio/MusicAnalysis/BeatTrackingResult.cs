namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Result of beat tracking.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> BeatTrackingResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
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
    /// Returns 0 if tempo is not available (0 or negative).
    /// </summary>
    public double AverageBeatInterval => Tempo > 0 ? 60.0 / Tempo : 0.0;
}
