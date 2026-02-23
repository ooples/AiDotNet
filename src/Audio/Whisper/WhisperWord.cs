namespace AiDotNet.Audio.Whisper;

/// <summary>
/// A transcribed word with timing information.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> WhisperWord provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class WhisperWord
{
    /// <summary>
    /// Gets or sets the word text.
    /// </summary>
    public string Word { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public double Confidence { get; set; }
}
