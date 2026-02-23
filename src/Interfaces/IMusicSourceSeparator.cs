namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for music source separation models that isolate individual instruments/vocals from a mix.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Music source separation (also called audio source separation or "unmixing") takes a
/// mixed audio signal and separates it into individual components like vocals, drums,
/// bass, and other instruments.
/// </para>
/// <para>
/// <b>For Beginners:</b> Source separation is like un-mixing a smoothie back into
/// its original fruits.
///
/// How it works:
/// 1. The mixed audio is converted to a spectrogram
/// 2. A neural network learns which parts belong to which source
/// 3. Masks are applied to isolate each source
/// 4. Individual spectrograms are converted back to audio
///
/// Common separations:
/// - 2-stem: Vocals vs Accompaniment
/// - 4-stem: Vocals, Drums, Bass, Other
/// - 5-stem: Vocals, Drums, Bass, Piano, Other
///
/// Use cases:
/// - Karaoke (remove vocals)
/// - Remixing (isolate and rearrange parts)
/// - Music transcription (analyze individual instruments)
/// - Sample extraction (get drum loops, vocal hooks)
/// - Music education (practice with isolated parts)
///
/// Popular models:
/// - Demucs (Facebook/Meta)
/// - Spleeter (Deezer)
/// - Open-Unmix
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MusicSourceSeparator")]
public interface IMusicSourceSeparator<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the sources this model can separate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common sources: "vocals", "drums", "bass", "other", "piano".
    /// </para>
    /// </remarks>
    IReadOnlyList<string> SupportedSources { get; }

    /// <summary>
    /// Gets the number of stems/sources this model produces.
    /// </summary>
    int NumStems { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Separates all sources from the audio mix.
    /// </summary>
    /// <param name="audio">Mixed audio waveform tensor [samples] or [channels, samples].</param>
    /// <returns>Separation result with all isolated sources.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for separating audio.
    /// - Pass in a mixed song
    /// - Get back individual tracks for each instrument/voice
    /// </para>
    /// </remarks>
    SourceSeparationResult<T> Separate(Tensor<T> audio);

    /// <summary>
    /// Separates all sources asynchronously.
    /// </summary>
    /// <param name="audio">Mixed audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Separation result with all isolated sources.</returns>
    Task<SourceSeparationResult<T>> SeparateAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Extracts a specific source from the mix.
    /// </summary>
    /// <param name="audio">Mixed audio waveform tensor.</param>
    /// <param name="source">The source to extract (e.g., "vocals", "drums").</param>
    /// <returns>Isolated audio for the requested source.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you only need one specific part.
    /// More efficient than separating everything if you only need vocals.
    /// </para>
    /// </remarks>
    Tensor<T> ExtractSource(Tensor<T> audio, string source);

    /// <summary>
    /// Removes a specific source from the mix.
    /// </summary>
    /// <param name="audio">Mixed audio waveform tensor.</param>
    /// <param name="source">The source to remove (e.g., "vocals" for karaoke).</param>
    /// <returns>Audio with the specified source removed.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This removes a source instead of extracting it.
    /// - RemoveSource("vocals") = karaoke track
    /// - RemoveSource("drums") = drumless practice track
    /// </para>
    /// </remarks>
    Tensor<T> RemoveSource(Tensor<T> audio, string source);

    /// <summary>
    /// Gets the soft mask for a specific source.
    /// </summary>
    /// <param name="audio">Mixed audio waveform tensor.</param>
    /// <param name="source">The source to get the mask for.</param>
    /// <returns>Soft mask tensor indicating how much of each time-frequency bin belongs to this source.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A soft mask shows the probability that each part
    /// of the audio belongs to a specific source. Values near 1 mean "definitely
    /// this source", values near 0 mean "definitely not this source".
    /// </para>
    /// </remarks>
    Tensor<T> GetSourceMask(Tensor<T> audio, string source);

    /// <summary>
    /// Remixes the separated sources with custom volumes.
    /// </summary>
    /// <param name="separationResult">Previous separation result.</param>
    /// <param name="sourceVolumes">Volume multipliers for each source (1.0 = original).</param>
    /// <returns>Remixed audio with adjusted source volumes.</returns>
    Tensor<T> Remix(SourceSeparationResult<T> separationResult, IReadOnlyDictionary<string, double> sourceVolumes);
}

/// <summary>
/// Result of source separation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SourceSeparationResult<T>
{
    /// <summary>
    /// Gets or sets the separated sources.
    /// </summary>
    public IReadOnlyDictionary<string, Tensor<T>> Sources { get; set; } = new Dictionary<string, Tensor<T>>();

    /// <summary>
    /// Gets or sets the original mixed audio.
    /// </summary>
    public Tensor<T> OriginalMix { get; set; } = default!;

    /// <summary>
    /// Gets or sets the sample rate of the audio.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the duration in seconds.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets quality metrics for the separation.
    /// </summary>
    public SeparationQuality<T>? Quality { get; set; }

    /// <summary>
    /// Gets a specific source by name.
    /// </summary>
    /// <param name="sourceName">The source name (e.g., "vocals").</param>
    /// <returns>The separated source audio.</returns>
    public Tensor<T> GetSource(string sourceName) =>
        Sources.TryGetValue(sourceName, out var source) ? source : throw new KeyNotFoundException($"Source '{sourceName}' not found.");
}

/// <summary>
/// Quality metrics for source separation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SeparationQuality<T>
{
    /// <summary>
    /// Gets or sets the Signal-to-Distortion Ratio (SDR) for each source.
    /// </summary>
    public IReadOnlyDictionary<string, T> SDR { get; set; } = new Dictionary<string, T>();

    /// <summary>
    /// Gets or sets the Signal-to-Interference Ratio (SIR) for each source.
    /// </summary>
    public IReadOnlyDictionary<string, T> SIR { get; set; } = new Dictionary<string, T>();

    /// <summary>
    /// Gets or sets the Signal-to-Artifacts Ratio (SAR) for each source.
    /// </summary>
    public IReadOnlyDictionary<string, T> SAR { get; set; } = new Dictionary<string, T>();
}
