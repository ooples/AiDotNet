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
    /// <summary>
    /// Source-file sample rate. NSynth audio is shipped at 16 kHz; this loader does
    /// not currently resample, so this property is informational and must equal 16000.
    /// </summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Samples per clip (4 sec * 16 kHz = 64,000).</summary>
    public int Samples { get; set; } = 16000 * 4;
    public int? MaxSamples { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        // SampleRate is currently informational — the loader passes raw 16 kHz samples through
        // AudioLoaderHelper without resampling. Reject mismatches rather than silently shipping
        // a wrong-rate waveform.
        if (SampleRate != 16000)
            throw new ArgumentOutOfRangeException(nameof(SampleRate),
                $"NSynth ships only at 16 kHz; resampling is not implemented in this loader. Got {SampleRate}.");
        if (Samples <= 0) throw new ArgumentOutOfRangeException(nameof(Samples), "Samples must be positive.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
    }
}
