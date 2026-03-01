using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Configuration options for the HMDB51 data loader.
/// </summary>
/// <remarks>
/// <para>
/// HMDB51 contains 6,766 video clips from 51 human action categories. Each category has
/// at least 101 clips. 3 train/test splits are provided.
/// </para>
/// </remarks>
public sealed class Hmdb51DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Number of frames to sample per video. Default is 16.</summary>
    public int FramesPerVideo { get; set; } = 16;
    /// <summary>Frame width. Default is 224.</summary>
    public int FrameWidth { get; set; } = 224;
    /// <summary>Frame height. Default is 224.</summary>
    public int FrameHeight { get; set; } = 224;
    /// <summary>Whether to normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;
    /// <summary>Split number (1, 2, or 3). Default is 1.</summary>
    public int SplitNumber { get; set; } = 1;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (FramesPerVideo <= 0) throw new ArgumentOutOfRangeException(nameof(FramesPerVideo), "FramesPerVideo must be positive.");
        if (FrameWidth <= 0) throw new ArgumentOutOfRangeException(nameof(FrameWidth), "FrameWidth must be positive.");
        if (FrameHeight <= 0) throw new ArgumentOutOfRangeException(nameof(FrameHeight), "FrameHeight must be positive.");
        if (SplitNumber < 1 || SplitNumber > 3) throw new ArgumentOutOfRangeException(nameof(SplitNumber), "SplitNumber must be between 1 and 3.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
        if (!Enum.IsDefined(typeof(Geometry.DatasetSplit), Split))
            throw new ArgumentOutOfRangeException(nameof(Split), "Split must be a valid DatasetSplit value.");
        if (DataPath is not null && string.IsNullOrWhiteSpace(DataPath))
            throw new ArgumentException("DataPath must not be empty or whitespace when provided.", nameof(DataPath));
    }
}
