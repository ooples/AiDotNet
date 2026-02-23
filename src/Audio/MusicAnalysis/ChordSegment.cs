namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Represents a chord segment in audio.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ChordSegment provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class ChordSegment
{
    /// <summary>
    /// Gets or sets the chord name (e.g., "C", "Am", "G7").
    /// </summary>
    public string Name { get; set; } = string.Empty;

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

    /// <summary>
    /// Gets or sets the confidence score (0-1).
    /// </summary>
    public double Confidence { get; set; }
}
