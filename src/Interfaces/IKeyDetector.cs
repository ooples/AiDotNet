namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for musical key detection models that identify the key and mode of music.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Key detection identifies the musical key (e.g., C major, A minor) of a piece of music.
/// The key defines the central note (tonic) and scale (major/minor) that the music is based on.
/// </para>
/// <para>
/// <b>For Beginners:</b> The musical key is like the "home base" of a song.
///
/// What is a key?
/// - Every song has a central note that feels like "home"
/// - The key tells you which note that is and whether it's major (happy) or minor (sad)
/// - "C major" means C is home and it sounds happy
/// - "A minor" means A is home and it sounds sad/dark
///
/// How key detection works:
/// 1. Audio is analyzed to find which notes are used most
/// 2. This is compared to key profiles (templates of note usage)
/// 3. The best-matching key is selected
///
/// Why it matters:
/// - DJ mixing (match keys for smooth transitions)
/// - Music recommendation (similar keys = similar feel)
/// - Music production (know what key to write melodies in)
/// - Transposition (shifting a song to a different key)
///
/// Related concepts:
/// - Relative keys: Am is the relative minor of C major (same notes)
/// - Parallel keys: C major and C minor (same root, different mode)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("KeyDetector")]
public interface IKeyDetector<T>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of keys this model can detect.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Typically 24 keys: 12 major + 12 minor.
    /// </para>
    /// </remarks>
    IReadOnlyList<string> SupportedKeys { get; }

    /// <summary>
    /// Detects the musical key of audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <returns>Key detection result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for finding the key.
    /// - Pass in audio of a song
    /// - Get back the key (e.g., "C major" or "A minor")
    /// </para>
    /// </remarks>
    KeyDetectionResult<T> Detect(Tensor<T> audio);

    /// <summary>
    /// Detects the musical key asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Key detection result.</returns>
    Task<KeyDetectionResult<T>> DetectAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets key probabilities for all possible keys.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Dictionary mapping key names to probability scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Some songs are ambiguous - they might sound like
    /// they're in C major or A minor (these share the same notes). This method
    /// shows the probability for each possible key.
    /// </para>
    /// </remarks>
    IReadOnlyDictionary<string, T> GetKeyProbabilities(Tensor<T> audio);

    /// <summary>
    /// Tracks key changes over time within a piece.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="segmentDuration">Duration of each analysis segment in seconds.</param>
    /// <returns>Key tracking result with key over time.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Some songs change key during the song (called modulation).
    /// This tracks those changes over time.
    /// </para>
    /// </remarks>
    KeyTrackingResult<T> TrackKeyChanges(Tensor<T> audio, double segmentDuration = 5.0);

    /// <summary>
    /// Gets the Camelot wheel notation for a key.
    /// </summary>
    /// <param name="key">The key to convert (e.g., "C major").</param>
    /// <returns>Camelot notation (e.g., "8B").</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Camelot wheel is a DJ tool that shows which
    /// keys mix well together. Adjacent numbers on the wheel = compatible keys.
    /// </para>
    /// </remarks>
    string GetCamelotNotation(string key);

    /// <summary>
    /// Finds compatible keys for mixing.
    /// </summary>
    /// <param name="key">The reference key.</param>
    /// <returns>List of compatible keys for harmonic mixing.</returns>
    IReadOnlyList<string> GetCompatibleKeys(string key);

    /// <summary>
    /// Gets the relative major/minor key.
    /// </summary>
    /// <param name="key">The key to find the relative for.</param>
    /// <returns>The relative key.</returns>
    string GetRelativeKey(string key);
}

/// <summary>
/// Result of key detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class KeyDetectionResult<T>
{
    /// <summary>
    /// Gets or sets the detected key (e.g., "C major", "A minor").
    /// </summary>
    public string Key { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the tonic note (e.g., "C", "A").
    /// </summary>
    public string Tonic { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the mode (major or minor).
    /// </summary>
    public MusicalMode Mode { get; set; } = MusicalMode.Major;

    /// <summary>
    /// Gets or sets the confidence score (0.0 to 1.0).
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the Camelot notation (e.g., "8B").
    /// </summary>
    public string CamelotNotation { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the Open Key notation (e.g., "1d").
    /// </summary>
    public string OpenKeyNotation { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the relative key.
    /// </summary>
    public string RelativeKey { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the parallel key.
    /// </summary>
    public string ParallelKey { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets alternative key hypotheses.
    /// </summary>
    public IReadOnlyList<KeyHypothesis<T>> Alternatives { get; set; } = Array.Empty<KeyHypothesis<T>>();
}

/// <summary>
/// Musical mode (major or minor).
/// </summary>
public enum MusicalMode
{
    /// <summary>Major mode (typically sounds happy/bright).</summary>
    Major,
    /// <summary>Minor mode (typically sounds sad/dark).</summary>
    Minor
}

/// <summary>
/// A key hypothesis with confidence score.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class KeyHypothesis<T>
{
    /// <summary>
    /// Gets or sets the key.
    /// </summary>
    public string Key { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;
}

/// <summary>
/// Result of key tracking over time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class KeyTrackingResult<T>
{
    /// <summary>
    /// Gets or sets the primary key of the piece.
    /// </summary>
    public string PrimaryKey { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets key segments over time.
    /// </summary>
    public IReadOnlyList<KeySegment<T>> Segments { get; set; } = Array.Empty<KeySegment<T>>();

    /// <summary>
    /// Gets or sets detected modulation points.
    /// </summary>
    public IReadOnlyList<ModulationPoint<T>> Modulations { get; set; } = Array.Empty<ModulationPoint<T>>();

    /// <summary>
    /// Gets or sets whether the piece has significant key changes.
    /// </summary>
    public bool HasKeyChanges { get; set; }
}

/// <summary>
/// A segment with a detected key.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class KeySegment<T>
{
    /// <summary>
    /// Gets or sets the key for this segment.
    /// </summary>
    public string Key { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;
}

/// <summary>
/// A point where the key changes (modulation).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ModulationPoint<T>
{
    /// <summary>
    /// Gets or sets the time of the modulation in seconds.
    /// </summary>
    public double Time { get; set; }

    /// <summary>
    /// Gets or sets the key before the modulation.
    /// </summary>
    public string FromKey { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the key after the modulation.
    /// </summary>
    public string ToKey { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence in the modulation detection.
    /// </summary>
    public T Confidence { get; set; } = default!;
}
