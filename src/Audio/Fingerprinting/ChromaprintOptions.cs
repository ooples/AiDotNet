using AiDotNet.Models.Options;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Configuration options for Chromaprint fingerprinting.
/// </summary>
public class ChromaprintOptions : ModelOptions
{
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
