namespace AiDotNet.Data.Video;

/// <summary>
/// Configuration options for the <see cref="VideoFrameDataset{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Videos are represented as directories of sequentially numbered image frames.
/// This is the standard format for preprocessed ML video datasets (UCF-101, Kinetics, etc.).
/// Structure: root/class_name/video_name/frame_001.bmp, frame_002.bmp, ...
/// </para>
/// </remarks>
public sealed class VideoFrameDatasetOptions
{
    /// <summary>
    /// Root directory containing class subdirectories, each with video subdirectories containing frame images.
    /// </summary>
    public string RootDirectory { get; set; } = string.Empty;

    /// <summary>
    /// File extensions for frame images. Default supports formats decoded by ImageHelper (BMP, PPM, PGM).
    /// </summary>
    public string[] FrameExtensions { get; set; } = new[] { ".bmp", ".ppm", ".pgm" };

    /// <summary>
    /// Number of frames to sample per video. Default is 16.
    /// Frames are sampled uniformly across the video duration.
    /// </summary>
    public int FramesPerVideo { get; set; } = 16;

    /// <summary>
    /// Target frame width in pixels. Default is 112.
    /// </summary>
    public int FrameWidth { get; set; } = 112;

    /// <summary>
    /// Target frame height in pixels. Default is 112.
    /// </summary>
    public int FrameHeight { get; set; } = 112;

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
    /// Whether class labels are determined by parent directory names. Default is true.
    /// </summary>
    public bool UseDirectoryLabels { get; set; } = true;
}
