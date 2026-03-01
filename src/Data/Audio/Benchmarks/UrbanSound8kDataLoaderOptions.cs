using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the UrbanSound8K data loader.
/// </summary>
/// <remarks>
/// <para>
/// UrbanSound8K contains 8,732 labeled sound excerpts (<=4s) of urban sounds in 10 classes:
/// air conditioner, car horn, children playing, dog bark, drilling, engine idling,
/// gun shot, jackhammer, siren, street music. 10 predefined cross-validation folds.
/// </para>
/// </remarks>
public sealed class UrbanSound8kDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 22050.</summary>
    public int SampleRate { get; set; } = 22050;
    /// <summary>Maximum audio duration in seconds. Default is 4.</summary>
    public double MaxDurationSeconds { get; set; } = 4.0;
    /// <summary>Cross-validation fold to use as test (1-10). Default is 10.</summary>
    public int TestFold { get; set; } = 10;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "Sample rate must be positive.");
        if (MaxDurationSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(MaxDurationSeconds), "Max duration must be positive.");
        if (TestFold < 1 || TestFold > 10) throw new ArgumentOutOfRangeException(nameof(TestFold), "TestFold must be between 1 and 10.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
