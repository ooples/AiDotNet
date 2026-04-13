using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the Google Speech Commands v2 data loader.
/// </summary>
/// <remarks>
/// <para>
/// Google Speech Commands v2 contains ~65,000 one-second audio clips of 35 spoken words
/// recorded by thousands of different speakers at 16kHz. The "core" 12-class subset
/// (yes, no, up, down, left, right, on, off, stop, go, silence, unknown) is the
/// standard benchmark, with CNN baselines achieving ~95% accuracy.
/// </para>
/// <para>
/// Dataset: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz (2.3GB)
/// Paper: https://arxiv.org/abs/1804.03209
/// License: Creative Commons BY 4.0
/// </para>
/// </remarks>
public sealed class SpeechCommandsDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>Automatically download if not present. Default is true.</summary>
    public bool AutoDownload { get; set; } = true;

    /// <summary>Optional maximum number of samples to load per class.</summary>
    public int? MaxSamplesPerClass { get; set; }

    /// <summary>Sample rate in Hz. Default is 16000 (native Speech Commands rate).</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Use the core 12-class subset (yes, no, up, down, left, right, on, off, stop, go,
    /// silence, unknown) instead of all 35 classes. Default is true.
    /// </summary>
    public bool UseCoreSubset { get; set; } = true;

    /// <summary>
    /// Target audio length in samples. Clips shorter than this are zero-padded;
    /// clips longer are truncated. Default is 16000 (1 second at 16kHz).
    /// </summary>
    public int TargetLength { get; set; } = 16000;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "Sample rate must be positive.");
        if (TargetLength <= 0) throw new ArgumentOutOfRangeException(nameof(TargetLength), "TargetLength must be positive.");
        if (MaxSamplesPerClass is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamplesPerClass), "MaxSamplesPerClass must be positive when specified.");
    }
}
