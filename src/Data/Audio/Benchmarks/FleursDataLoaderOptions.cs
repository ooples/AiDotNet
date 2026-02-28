using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the FLEURS data loader.
/// </summary>
/// <remarks>
/// <para>
/// FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a
/// multilingual speech benchmark covering 102 languages with ~12 hours per language.
/// Each utterance has a text transcription.
/// </para>
/// </remarks>
public sealed class FleursDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 16000.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Maximum audio duration in seconds. Default is 15.</summary>
    public double MaxDurationSeconds { get; set; } = 15.0;
    /// <summary>Language code. Default is "en_us".</summary>
    public string Language { get; set; } = "en_us";
}
