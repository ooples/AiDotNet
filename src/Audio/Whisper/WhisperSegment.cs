namespace AiDotNet.Audio.Whisper;

/// <summary>
/// A segment of transcribed speech with timing.
/// </summary>
public class WhisperSegment
{
    /// <summary>
    /// Gets or sets the segment text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the words in this segment.
    /// </summary>
    public List<WhisperWord> Words { get; set; } = [];
}
