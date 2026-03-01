using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the COCO Detection data loader.
/// </summary>
/// <remarks>
/// <para>
/// COCO (Common Objects in Context) 2017 Detection contains 118K training and 5K validation images
/// with 80 object categories. Annotations include bounding boxes, segmentation masks, and captions.
/// This loader focuses on the object detection task (bounding boxes + class labels).
/// </para>
/// </remarks>
public sealed class CocoDetectionDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 640.</summary>
    public int ImageSize { get; set; } = 640;

    /// <summary>
    /// Maximum number of detections per image. Detections are zero-padded to this size.
    /// Default is 100.
    /// </summary>
    public int MaxDetections { get; set; } = 100;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(ImageSize), "ImageSize must be positive.");
        if (MaxDetections <= 0) throw new ArgumentOutOfRangeException(nameof(MaxDetections), "MaxDetections must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
