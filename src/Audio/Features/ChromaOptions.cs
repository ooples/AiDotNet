namespace AiDotNet.Audio.Features;

/// <summary>
/// Options for chroma feature extraction.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Chroma model. Default values follow the original paper settings.</para>
/// </remarks>
public class ChromaOptions : AudioFeatureOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ChromaOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ChromaOptions(ChromaOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Normalize = other.Normalize;
        TuningFrequency = other.TuningFrequency;
        NumOctaves = other.NumOctaves;
    }

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
