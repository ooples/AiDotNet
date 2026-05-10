using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the GTZAN music genre classification loader.
/// </summary>
/// <remarks>
/// <para>
/// GTZAN (Tzanetakis &amp; Cook 2002) — 10 genres × 100 30-second clips each
/// = 1,000 mono WAV files at 22,050 Hz. The canonical music genre
/// classification benchmark, despite known label noise. Useful for
/// MIR research entry-level evaluation.
/// </para>
/// </remarks>
public sealed class GtzanDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    /// <summary>Sample rate for resampling. Default 22050 (matches the source).</summary>
    public int SampleRate { get; set; } = 22050;
    /// <summary>Number of samples per clip. Default 30 sec * 22050 Hz = 661,500. Clips are zero-padded to this length.</summary>
    public int Samples { get; set; } = 30 * 22050;
    /// <summary>Train/test split fraction (per-class deterministic). Default 0.8 → 80 train / 20 test per class.</summary>
    public double TrainFraction { get; set; } = 0.8;
    public int? MaxSamples { get; set; }
}
