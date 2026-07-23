using AiDotNet.Data.Geometry;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Configuration options for the Mozilla Common Voice data loader.
/// </summary>
/// <remarks>
/// <para>
/// Common Voice is a multilingual speech corpus with ~19K+ hours across 100+ languages.
/// Audio is stored as MP3 files. For use with this loader, pre-convert to WAV format
/// (e.g., using ffmpeg: <c>ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav</c>).
/// </para>
/// </remarks>
public sealed class CommonVoiceDataLoaderOptions
{
    /// <summary>Dataset split to load. Default is Train.</summary>
    public DatasetSplit Split { get; set; } = DatasetSplit.Train;
    /// <summary>Root data path. When null, uses default cache path.</summary>
    public string? DataPath { get; set; }
    /// <summary>Automatically download if not present. Default is false (requires Mozilla agreement).</summary>
    public bool AutoDownload { get; set; }
    /// <summary>Optional maximum number of samples to load.</summary>
    public int? MaxSamples { get; set; }
    /// <summary>Sample rate in Hz. Default is 16000.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Maximum audio duration in seconds. Default is 10.</summary>
    public double MaxDurationSeconds { get; set; } = 10.0;
    /// <summary>Language code (e.g., "en", "fr", "de"). Default is "en".</summary>
    public string Language { get; set; } = "en";

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate), "SampleRate must be positive.");
        if (MaxDurationSeconds <= 0 || double.IsNaN(MaxDurationSeconds) || double.IsInfinity(MaxDurationSeconds))
            throw new ArgumentOutOfRangeException(nameof(MaxDurationSeconds), "MaxDurationSeconds must be a positive finite number.");
        if (MaxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(MaxSamples), "MaxSamples must be positive when specified.");
        if (string.IsNullOrWhiteSpace(Language)) throw new ArgumentException("Language must not be empty or whitespace.", nameof(Language));
        if (DataPath is not null && string.IsNullOrWhiteSpace(DataPath))
            throw new ArgumentException("DataPath must not be empty or whitespace when provided.", nameof(DataPath));
        if (!Enum.IsDefined(typeof(Geometry.DatasetSplit), Split))
            throw new ArgumentOutOfRangeException(nameof(Split), "Split must be a valid DatasetSplit value.");
    }
}
