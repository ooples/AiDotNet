using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Cityscapes semantic segmentation loader.
/// </summary>
/// <remarks>
/// <para>
/// Cityscapes is the canonical urban-driving semantic segmentation
/// benchmark (Cordts et al. 2016). The "fine" annotations have 5,000
/// pixel-accurate labelled images at 2048×1024 across 50 cities. 19
/// evaluation classes (out of 30 source classes; 11 are reserved/ignored).
/// </para>
/// <para>
/// <b>Auto-download is disabled by default</b> — Cityscapes requires
/// account sign-up at cityscapes-dataset.com. Download the two archives
/// manually and extract under <see cref="DataPath"/>:
/// <c>leftImg8bit_trainvaltest.zip</c> and <c>gtFine_trainvaltest.zip</c>.
/// </para>
/// </remarks>
public sealed class CityscapesDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    /// <summary>Auto-download is OFF by default — Cityscapes requires manual sign-up.</summary>
    public bool AutoDownload { get; set; } = false;
    public int ImageHeight { get; set; } = 512;
    public int ImageWidth { get; set; } = 1024;
    public bool Normalize { get; set; } = true;
    /// <summary>Map the 30 source classes to the 19 evaluation classes (CityscapesScripts ID2trainID). Default true.</summary>
    public bool MapToTrainIds { get; set; } = true;
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (ImageHeight <= 0) throw new ArgumentOutOfRangeException(nameof(ImageHeight), "ImageHeight must be positive.");
        if (ImageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(ImageWidth), "ImageWidth must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
