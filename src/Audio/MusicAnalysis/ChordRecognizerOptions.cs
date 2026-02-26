using AiDotNet.Models.Options;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for chord recognition.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ChordRecognizer model. Default values follow the original paper settings.</para>
/// </remarks>
public class ChordRecognizerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ChordRecognizerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ChordRecognizerOptions(ChordRecognizerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        MinChromaEnergy = other.MinChromaEnergy;
        MinConfidence = other.MinConfidence;
        MinSegmentDuration = other.MinSegmentDuration;
    }

    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the minimum chroma energy to consider.
    /// </summary>
    public double MinChromaEnergy { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the minimum confidence for chord detection.
    /// </summary>
    public double MinConfidence { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum segment duration in seconds.
    /// </summary>
    public double MinSegmentDuration { get; set; } = 0.2;
}
