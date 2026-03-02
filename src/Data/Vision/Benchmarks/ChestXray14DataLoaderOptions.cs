using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the NIH Chest X-ray 14 data loader.
/// </summary>
/// <remarks>
/// <para>
/// NIH Chest X-ray 14 contains 112,120 frontal-view chest X-ray images from 30,805 unique patients
/// with 14 disease labels mined from radiology reports using NLP. This is a multi-label classification
/// task where each image can have zero or more of the 14 disease labels.
/// </para>
/// </remarks>
public sealed class ChestXray14DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (dataset is ~45GB, requires NIH agreement).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 224.</summary>
    public int ImageSize { get; set; } = 224;
}
