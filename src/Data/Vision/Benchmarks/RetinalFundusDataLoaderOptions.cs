using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Configuration options for the Retinal Fundus data loader.
/// </summary>
/// <remarks>
/// <para>
/// Retinal fundus photography datasets are used for diabetic retinopathy and glaucoma detection.
/// This loader supports common retinal datasets such as EyePACS/Kaggle Diabetic Retinopathy
/// (35K train / 53K test) with 5-class severity grading (0: No DR, 1: Mild, 2: Moderate,
/// 3: Severe, 4: Proliferative DR).
/// </para>
/// </remarks>
public sealed class RetinalFundusDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download if not present. Default is false (requires Kaggle agreement).
    /// </summary>
    public bool AutoDownload { get; set; }

    /// <summary>Normalize pixel values to [0, 1]. Default is true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }

    /// <summary>Target image size. Default is 512.</summary>
    public int ImageSize { get; set; } = 512;
}
