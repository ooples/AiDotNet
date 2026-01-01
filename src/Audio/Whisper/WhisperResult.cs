namespace AiDotNet.Audio.Whisper;

/// <summary>
/// Result of Whisper transcription.
/// </summary>
public class WhisperResult
{
    /// <summary>
    /// Gets or sets the transcribed text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the detected language code.
    /// </summary>
    public string? DetectedLanguage { get; set; }

    /// <summary>
    /// Gets or sets the language detection probability.
    /// </summary>
    public double LanguageProbability { get; set; }

    /// <summary>
    /// Gets or sets the word-level timestamps if requested.
    /// </summary>
    public List<WhisperWord> Words { get; set; } = [];

    /// <summary>
    /// Gets or sets the segment-level timestamps.
    /// </summary>
    public List<WhisperSegment> Segments { get; set; } = [];

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }
}
