using AiDotNet.Models.Options;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Configuration options for Chromaprint fingerprinting.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Chromaprint model. Default values follow the original paper settings.</para>
/// </remarks>
public class ChromaprintOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ChromaprintOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ChromaprintOptions(ChromaprintOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        ContextSize = other.ContextSize;
        HashStep = other.HashStep;
        MaxBitDifference = other.MaxBitDifference;
    }

    /// <summary>
    /// Gets or sets the sample rate (default 11025 Hz for efficiency).
    /// </summary>
    public int SampleRate { get; set; } = 11025;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the number of context frames for hashing.
    /// </summary>
    public int ContextSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the step between hash computations.
    /// </summary>
    public int HashStep { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum bit difference for approximate matching.
    /// </summary>
    public int MaxBitDifference { get; set; } = 2;
}
