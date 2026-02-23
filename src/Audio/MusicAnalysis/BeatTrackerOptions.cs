using AiDotNet.Models.Options;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for beat tracking.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the BeatTracker model. Default values follow the original paper settings.</para>
/// </remarks>
public class BeatTrackerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public BeatTrackerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BeatTrackerOptions(BeatTrackerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        MinTempo = other.MinTempo;
        MaxTempo = other.MaxTempo;
        SmoothingWindow = other.SmoothingWindow;
        TempoFlexibility = other.TempoFlexibility;
    }

    /// <summary>
    /// Gets or sets the sample rate of the audio.
    /// </summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>
    /// Gets or sets the FFT size for spectral analysis.
    /// </summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the hop length for frame extraction.
    /// </summary>
    public int HopLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the minimum tempo to consider (BPM).
    /// </summary>
    public double MinTempo { get; set; } = 60.0;

    /// <summary>
    /// Gets or sets the maximum tempo to consider (BPM).
    /// </summary>
    public double MaxTempo { get; set; } = 200.0;

    /// <summary>
    /// Gets or sets the smoothing window size for onset envelope.
    /// </summary>
    public int SmoothingWindow { get; set; } = 3;

    /// <summary>
    /// Gets or sets how flexible the tempo tracking is (0-1).
    /// </summary>
    public double TempoFlexibility { get; set; } = 0.2;
}
