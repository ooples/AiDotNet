namespace AiDotNet.Data.Video;

/// <summary>
/// Configuration options for the <see cref="VideoFrameDataset{T}"/>.
/// </summary>
public sealed class VideoFrameDatasetOptions
{
    /// <summary>
    /// Root directory containing video files or class subdirectories.
    /// </summary>
    public string RootDirectory { get; set; } = string.Empty;

    /// <summary>
    /// File extensions to include. Default is common video formats.
    /// </summary>
    public string[] Extensions { get; set; } = new[] { ".mp4", ".avi", ".mkv", ".mov", ".wmv" };

    /// <summary>
    /// Number of frames to extract per video. Default is 16.
    /// </summary>
    public int FramesPerVideo { get; set; } = 16;

    /// <summary>
    /// Target frame width in pixels. Default is 224.
    /// </summary>
    public int FrameWidth { get; set; } = 224;

    /// <summary>
    /// Target frame height in pixels. Default is 224.
    /// </summary>
    public int FrameHeight { get; set; } = 224;

    /// <summary>
    /// Number of color channels per frame. Default is 3 (RGB).
    /// </summary>
    public int Channels { get; set; } = 3;

    /// <summary>
    /// Whether to normalize pixel values to [0, 1]. Default is true.
    /// </summary>
    public bool NormalizePixels { get; set; } = true;

    /// <summary>
    /// Optional maximum number of videos to load.
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
