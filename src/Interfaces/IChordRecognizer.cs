namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for chord recognition models that identify musical chords in audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chord recognition analyzes audio to identify the musical chords being played.
/// This involves detecting the simultaneous notes and classifying them into
/// standard chord types (major, minor, seventh, etc.).
/// </para>
/// <para>
/// <b>For Beginners:</b> Chord recognition is like having a musician listen to a song
/// and tell you what chords are being played.
///
/// How it works:
/// 1. Audio is converted to a chromagram (12 pitch classes)
/// 2. The pitch content is compared to known chord templates
/// 3. The best-matching chord is selected for each time frame
///
/// What are chords?
/// - A chord is multiple notes played together
/// - "C major" = C + E + G notes together
/// - "A minor" = A + C + E notes together
/// - The chord creates the harmony of the music
///
/// Common use cases:
/// - Learning songs (getting chord charts automatically)
/// - Music production (analyzing harmony)
/// - Music generation (understanding structure)
/// - Cover song detection (comparing harmonic content)
/// </para>
/// </remarks>
public interface IChordRecognizer<T>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of chord types this model can recognize.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common chord types: Major, Minor, Diminished, Augmented, 7th, etc.
    /// </para>
    /// </remarks>
    IReadOnlyList<string> SupportedChordTypes { get; }

    /// <summary>
    /// Gets the time resolution for chord detection in seconds.
    /// </summary>
    double TimeResolution { get; }

    /// <summary>
    /// Recognizes chords in audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <returns>Chord recognition result with chord sequence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for finding chords.
    /// - Pass in audio of music
    /// - Get back a list of chords and when they occur
    /// </para>
    /// </remarks>
    ChordRecognitionResult<T> Recognize(Tensor<T> audio);

    /// <summary>
    /// Recognizes chords asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Chord recognition result.</returns>
    Task<ChordRecognitionResult<T>> RecognizeAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets chord probabilities for each time frame.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tensor of chord probabilities [time_frames, num_chords].</returns>
    /// <remarks>
    /// <para>
    /// Useful for visualizing chord likelihood over time or for
    /// custom post-processing of chord detection.
    /// </para>
    /// </remarks>
    Tensor<T> GetChordProbabilities(Tensor<T> audio);

    /// <summary>
    /// Extracts chromagram features from audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Chromagram tensor [12, time_frames] with energy per pitch class.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A chromagram shows how much energy is in each of the
    /// 12 musical notes (C, C#, D, etc.) over time. It's the raw data used for
    /// chord recognition.
    /// </para>
    /// </remarks>
    Tensor<T> ExtractChromagram(Tensor<T> audio);

    /// <summary>
    /// Converts a chord symbol to its component notes.
    /// </summary>
    /// <param name="chordSymbol">Chord symbol (e.g., "Cmaj", "Am7").</param>
    /// <returns>List of note names in the chord.</returns>
    IReadOnlyList<string> GetChordNotes(string chordSymbol);
}

/// <summary>
/// Result of chord recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ChordRecognitionResult<T>
{
    /// <summary>
    /// Gets or sets the detected chord segments.
    /// </summary>
    public IReadOnlyList<ChordSegment<T>> Segments { get; set; } = Array.Empty<ChordSegment<T>>();

    /// <summary>
    /// Gets or sets the total duration in seconds.
    /// </summary>
    public double TotalDuration { get; set; }

    /// <summary>
    /// Gets the unique chords used in the song.
    /// </summary>
    public IReadOnlyList<string> UniqueChords { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets chord usage statistics.
    /// </summary>
    public IReadOnlyDictionary<string, ChordStatistics<T>> ChordStats { get; set; } =
        new Dictionary<string, ChordStatistics<T>>();
}

/// <summary>
/// A segment of audio with a detected chord.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ChordSegment<T>
{
    /// <summary>
    /// Gets or sets the chord symbol (e.g., "C", "Am", "G7").
    /// </summary>
    public string Chord { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the root note of the chord.
    /// </summary>
    public string Root { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the chord quality (e.g., "major", "minor", "dominant7").
    /// </summary>
    public string Quality { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the confidence score for this chord.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets the duration of this segment.
    /// </summary>
    public double Duration => EndTime - StartTime;
}

/// <summary>
/// Statistics for a chord in the recognition result.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ChordStatistics<T>
{
    /// <summary>
    /// Gets or sets the total duration this chord was detected.
    /// </summary>
    public double TotalDuration { get; set; }

    /// <summary>
    /// Gets or sets the number of occurrences.
    /// </summary>
    public int NumOccurrences { get; set; }

    /// <summary>
    /// Gets or sets the percentage of total audio time.
    /// </summary>
    public double Percentage { get; set; }

    /// <summary>
    /// Gets or sets the average confidence when this chord was detected.
    /// </summary>
    public T AverageConfidence { get; set; } = default!;
}
