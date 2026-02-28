using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the FSD50K data loader.
/// </summary>
/// <remarks>
/// <para>
/// FSD50K (Freesound Dataset 50K) contains 51,197 audio clips with 200 sound event classes
/// from Freesound. Multi-label classification with hierarchical AudioSet ontology labels.
/// Audio clips range from 0.3s to 30s.
/// </para>
/// </remarks>
public sealed class Fsd50kDataLoaderOptions
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
    /// <summary>Maximum audio duration in seconds. Default is 10.</summary>
    public double MaxDurationSeconds { get; set; } = 10.0;
}
