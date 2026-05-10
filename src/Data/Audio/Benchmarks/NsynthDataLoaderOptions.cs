using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration for the NSynth (Neural Synth) audio dataset loader (Engel et al. 2017).
/// </summary>
/// <remarks>
/// <para>
/// NSynth — 305,979 musical notes from 1,006 instruments, each as a 4-second
/// 16 kHz mono WAV. Annotated with pitch, velocity, instrument family (11 classes),
/// instrument source (3 classes), and qualities. Standard benchmark for
/// audio synthesis / pitch-conditional generation.
/// </para>
/// </remarks>
public sealed class NsynthDataLoaderOptions
{
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    public string? DataPath { get; set; }
    public bool AutoDownload { get; set; } = true;
    /// <summary>Sample rate. Default 16000 (matches source).</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Samples per clip (4 sec * 16 kHz = 64,000).</summary>
    public int Samples { get; set; } = 16000 * 4;
    public int? MaxSamples { get; set; }
}
