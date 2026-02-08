using AiDotNet.Models.Options;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for chord recognition.
/// </summary>
public class ChordRecognizerOptions : ModelOptions
{
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
