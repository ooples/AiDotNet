namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Configuration options for spectrogram fingerprinting.
/// </summary>
public class SpectrogramFingerprintOptions
{
    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 11025;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the peak detection neighborhood size.
    /// </summary>
    public int PeakNeighborhood { get; set; } = 3;

    /// <summary>
    /// Gets or sets the minimum peak magnitude threshold.
    /// </summary>
    public double PeakThreshold { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum peaks per analysis window.
    /// </summary>
    public int MaxPeaksPerWindow { get; set; } = 5;

    /// <summary>
    /// Gets or sets the window size in frames for peak selection.
    /// Peaks are selected from non-overlapping windows of this size.
    /// </summary>
    public int PeakWindowSizeFrames { get; set; } = 20;

    /// <summary>
    /// Gets or sets the target zone start (frames ahead).
    /// </summary>
    public int TargetZoneStart { get; set; } = 1;

    /// <summary>
    /// Gets or sets the target zone end (frames ahead).
    /// </summary>
    public int TargetZoneEnd { get; set; } = 64;
}
