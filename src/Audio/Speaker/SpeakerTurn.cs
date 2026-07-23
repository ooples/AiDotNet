namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Represents a speaker turn in diarization output.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SpeakerTurn provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeakerTurn
{
    /// <summary>
    /// Gets or sets the speaker ID label.
    /// </summary>
    public string SpeakerId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the speaker index (0-based).
    /// </summary>
    public int SpeakerIndex { get; set; }

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets the duration in seconds.
    /// </summary>
    public double Duration => EndTime - StartTime;
}
