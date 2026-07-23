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
/// Dataset: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz (2.3GB)
/// Paper: https://arxiv.org/abs/1804.03209
/// License: Creative Commons BY 4.0
/// </para>
/// </remarks>
public sealed class SpeechCommandsDataLoaderOptions
{
    /// <summary>Native sample rate of the Speech Commands corpus (16kHz).</summary>
    public const int NativeSampleRate = 16000;

    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;

    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }

    /// <summary>
    /// Automatically download and extract the dataset if it isn't present at
    /// <see cref="DataPath"/>. Default is true. When false, missing data raises
    /// an <see cref="InvalidOperationException"/> instead.
    /// </summary>
    public bool AutoDownload { get; set; } = true;

    /// <summary>Optional maximum number of samples to load per class.</summary>
    public int? MaxSamplesPerClass { get; set; }

    /// <summary>
    /// Target audio sample rate in Hz. Default is 16000 (the Speech Commands native rate).
    /// When this differs from <see cref="NativeSampleRate"/> the loader resamples each
    /// clip via linear interpolation before truncation/padding to <see cref="TargetLength"/>.
    /// </summary>
    public int SampleRate { get; set; } = NativeSampleRate;

    /// <summary>
    /// Use the core 12-class subset (yes, no, up, down, left, right, on, off, stop, go,
    /// <c>_silence_</c>, <c>_unknown_</c>) instead of all 35 classes. Default is true.
    /// </summary>
    /// <remarks>
    /// The two synthetic classes follow the Warden 2018 benchmark spec: <c>_silence_</c>
    /// is sampled from the dataset's <c>_background_noise_/</c> directory (which is
    /// required when <see cref="SilenceSampleCount"/> is positive — the loader fails
    /// fast with <see cref="InvalidOperationException"/> if the directory is missing
    /// or empty, since a zero-filled silence class would silently create a different
    /// benchmark than the spec describes). <c>_unknown_</c> collapses every non-core
    /// word directory so the classifier learns an "out-of-vocabulary" bucket.
    /// </remarks>
    public bool UseCoreSubset { get; set; } = true;

    /// <summary>
    /// Target audio length in samples at <see cref="SampleRate"/>. Clips shorter than
    /// this are zero-padded; clips longer are truncated. Default is 16000 (1 second at
    /// the native 16kHz rate). When you change <see cref="SampleRate"/>, also adjust
    /// this value to keep the per-clip duration consistent (e.g. 8000 for 1 second
    /// at 8kHz).
    /// </summary>
    public int TargetLength { get; set; } = 16000;

    /// <summary>
    /// Train-equivalent number of <c>_silence_</c> clips to synthesize from the
    /// background-noise directory when the core 12-class subset is in use. Default
    /// is 2300, which matches the per-class ratio in the official 12-class training
    /// split. The loader scales the emitted silence count by <see cref="Split"/>
    /// (training ≈ full count, validation ≈ 12%, test ≈ 6%), so Validation/Test
    /// produce proportionally fewer silence samples than this configured value.
    /// Ignored when <see cref="UseCoreSubset"/> is false.
    /// </summary>
    public int SilenceSampleCount { get; set; } = 2300;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    /// <exception cref="ArgumentException">Thrown when <see cref="DataPath"/> is the empty string.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "Sample rate must be positive.");
        if (TargetLength <= 0) throw new ArgumentOutOfRangeException(nameof(TargetLength), "TargetLength must be positive.");
        if (MaxSamplesPerClass is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamplesPerClass), "MaxSamplesPerClass must be positive when specified.");
        if (SilenceSampleCount < 0) throw new ArgumentOutOfRangeException(nameof(SilenceSampleCount), "SilenceSampleCount must be non-negative.");
        // null means "use the default cache path" (set up by the loader constructor).
        // Empty/whitespace would be silently treated as the current working directory,
        // which is almost never what the caller intended.
        if (DataPath is not null && string.IsNullOrWhiteSpace(DataPath))
            throw new ArgumentException("DataPath must be null (for the default cache) or a non-empty, non-whitespace path.", nameof(DataPath));
    }
}
