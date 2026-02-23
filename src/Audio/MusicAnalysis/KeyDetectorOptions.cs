using AiDotNet.Models.Options;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Configuration options for key detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KeyDetector model. Default values follow the original paper settings.</para>
/// </remarks>
public class KeyDetectorOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public KeyDetectorOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KeyDetectorOptions(KeyDetectorOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
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
}
