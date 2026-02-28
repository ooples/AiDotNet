using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the GigaSpeech data loader.
/// </summary>
/// <remarks>
/// <para>
/// GigaSpeech is a multi-domain English ASR dataset with 10,000 hours of labeled audio
/// from audiobooks, podcasts, and YouTube. Subsets: XS (10h), S (250h), M (1000h), L (2500h), XL (10000h).
/// </para>
/// </remarks>
public sealed class GigaSpeechDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false.</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 16000.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Maximum audio duration in seconds. Default is 30.</summary>
    public double MaxDurationSeconds { get; set; } = 30.0;
    /// <summary>Subset to load. Default is "XS". Options: "XS", "S", "M", "L", "XL".</summary>
    public string Subset { get; set; } = "XS";
}
