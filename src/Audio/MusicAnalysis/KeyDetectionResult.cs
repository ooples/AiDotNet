namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Result of key detection.
/// </summary>
public class KeyDetectionResult
{
    /// <summary>
    /// Gets or sets the key index (0 = C, 1 = C#, etc.).
    /// </summary>
    public int KeyIndex { get; set; }

    /// <summary>
    /// Gets or sets the full key name (e.g., "C major", "A minor").
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the root note name (e.g., "C", "A").
    /// </summary>
    public string RootNote { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the key mode (major or minor).
    /// </summary>
    public KeyMode Mode { get; set; }

    /// <summary>
    /// Gets or sets the Pearson correlation with the key profile (-1 to 1).
    /// </summary>
    public double Correlation { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (0-1).
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Gets or sets the relative major/minor key.
    /// </summary>
    public string RelativeKey { get; set; } = string.Empty;
}
