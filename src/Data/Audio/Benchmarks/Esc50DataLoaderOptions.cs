using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the ESC-50 data loader.
/// </summary>
/// <remarks>
/// <para>
/// ESC-50 (Environmental Sound Classification) contains 2,000 5-second environmental audio
/// recordings organized into 50 classes (e.g., dog bark, rain, clock tick) with 40 clips per class.
/// 5 predefined cross-validation folds.
/// </para>
/// </remarks>
public sealed class Esc50DataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 44100.</summary>
    public int SampleRate { get; set; } = 44100;
    /// <summary>Cross-validation fold to use as test (1-5). Default is 5.</summary>
    public int TestFold { get; set; } = 5;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "Sample rate must be positive.");
        if (TestFold < 1 || TestFold > 5) throw new ArgumentOutOfRangeException(nameof(TestFold), "TestFold must be between 1 and 5.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
