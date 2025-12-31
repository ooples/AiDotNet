namespace AiDotNet.Audio.Features;

/// <summary>
/// Options for chroma feature extraction.
/// </summary>
public class ChromaOptions : AudioFeatureOptions
{
    /// <summary>
    /// Gets or sets whether to L2-normalize each chroma frame.
    /// Default is true.
    /// </summary>
    public bool Normalize { get; set; } = true;

    /// <summary>
    /// Gets or sets the tuning frequency for A4.
    /// Default is 440 Hz (standard concert pitch).
    /// </summary>
    public double TuningFrequency { get; set; } = 440.0;

    /// <summary>
    /// Gets or sets the number of octaves to include.
    /// Default is 7 (piano range).
    /// </summary>
    public int NumOctaves { get; set; } = 7;
}
