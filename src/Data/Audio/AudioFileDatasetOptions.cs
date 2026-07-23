namespace AiDotNet.Data.Audio;

/// <summary>
/// Configuration options for the <see cref="AudioFileDataset{T}"/>.
/// </summary>
public sealed class AudioFileDatasetOptions
{
    /// <summary>
    /// Root directory containing audio files or class subdirectories.
    /// </summary>
    public string RootDirectory { get; set; } = string.Empty;

    /// <summary>
    /// File extensions to include. Default is common audio formats.
    /// </summary>
    public string[] Extensions { get; set; } = new[] { ".wav", ".pcm", ".raw" };

    /// <summary>
    /// Target sample rate in Hz. Audio will be resampled to this rate.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Maximum duration in seconds. Audio will be truncated or padded to this length.
    /// </summary>
    public double DurationSeconds { get; set; } = 5.0;

    /// <summary>
    /// Whether to convert to mono. Default is true.
    /// </summary>
    public bool Mono { get; set; } = true;

    /// <summary>
    /// Whether to normalize audio values to [-1, 1]. Default is true.
    /// </summary>
    public bool Normalize { get; set; } = true;

    /// <summary>
    /// Optional maximum number of samples to load.
    /// </summary>
    public int? MaxSamples { get; set; }

    /// <summary>
    /// Optional random seed for reproducible sampling.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Whether class labels are determined by subdirectory names. Default is true.
    /// </summary>
    public bool UseDirectoryLabels { get; set; } = true;
}
