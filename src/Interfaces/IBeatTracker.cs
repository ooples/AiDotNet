namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for beat tracking models that detect tempo and beat positions in audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Beat tracking analyzes audio to find the rhythmic pulse (beats) and estimate the tempo
/// (beats per minute). This is fundamental to music information retrieval and enables
/// beat-synchronized processing like auto-DJ mixing and rhythmic visualization.
/// </para>
/// <para>
/// <b>For Beginners:</b> Beat tracking is like tapping your foot to music - finding the pulse.
///
/// How it works:
/// 1. Audio is analyzed for rhythmic events (drum hits, bass notes, etc.)
/// 2. Periodicity detection finds the most likely beat period
/// 3. Beat positions are refined to align with actual events
///
/// Common use cases:
/// - Music tempo detection ("this song is 120 BPM")
/// - DJ software (beat matching between songs)
/// - Music games (rhythm games like Guitar Hero)
/// - Audio visualization (beat-synced lights)
/// - Music production (quantizing to the beat)
///
/// Key concepts:
/// - BPM (Beats Per Minute): The tempo or speed of the music
/// - Downbeat: The first beat of a measure (often emphasized)
/// - Beat phase: Where in the beat cycle we are
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("BeatTracker")]
public interface IBeatTracker<T>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the minimum detectable BPM.
    /// </summary>
    double MinBPM { get; }

    /// <summary>
    /// Gets the maximum detectable BPM.
    /// </summary>
    double MaxBPM { get; }

    /// <summary>
    /// Detects tempo and beat positions in audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <returns>Beat tracking result with tempo and beat positions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for finding beats.
    /// - Pass in audio of a song
    /// - Get back the tempo (BPM) and when each beat occurs
    /// </para>
    /// </remarks>
    BeatTrackingResult<T> Track(Tensor<T> audio);

    /// <summary>
    /// Detects tempo and beat positions asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Beat tracking result.</returns>
    Task<BeatTrackingResult<T>> TrackAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Estimates tempo without detecting individual beat positions.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Estimated tempo in BPM.</returns>
    /// <remarks>
    /// <para>
    /// Faster than full beat tracking when you only need the tempo.
    /// </para>
    /// </remarks>
    T EstimateTempo(Tensor<T> audio);

    /// <summary>
    /// Gets multiple tempo hypotheses with confidence scores.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="numHypotheses">Number of hypotheses to return.</param>
    /// <returns>List of tempo hypotheses with confidence scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sometimes tempo is ambiguous (could be 60 or 120 BPM).
    /// This returns multiple possibilities with confidence scores.
    /// </para>
    /// </remarks>
    IReadOnlyList<TempoHypothesis<T>> GetTempoHypotheses(Tensor<T> audio, int numHypotheses = 5);

    /// <summary>
    /// Detects downbeat positions (first beat of each measure).
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="beatTrackingResult">Optional pre-computed beat tracking result.</param>
    /// <returns>Downbeat detection result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Downbeats are the "strong" beats (the "1" in counting "1-2-3-4").
    /// Finding downbeats helps identify the musical structure.
    /// </para>
    /// </remarks>
    DownbeatResult<T> DetectDownbeats(Tensor<T> audio, BeatTrackingResult<T>? beatTrackingResult = null);

    /// <summary>
    /// Computes onset strength envelope for visualization or custom processing.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Onset strength values over time.</returns>
    Tensor<T> ComputeOnsetStrength(Tensor<T> audio);
}

/// <summary>
/// Result of beat tracking.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class BeatTrackingResult<T>
{
    /// <summary>
    /// Gets or sets the estimated tempo in beats per minute.
    /// </summary>
    public T Tempo { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence in the tempo estimate (0.0 to 1.0).
    /// </summary>
    public T TempoConfidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the beat positions in seconds.
    /// </summary>
    public IReadOnlyList<double> BeatTimes { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the confidence score for each beat.
    /// </summary>
    public IReadOnlyList<T> BeatConfidences { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the beat interval in seconds.
    /// </summary>
    public double BeatInterval { get; set; }

    /// <summary>
    /// Gets the number of detected beats.
    /// </summary>
    public int NumBeats => BeatTimes.Count;
}

/// <summary>
/// A tempo hypothesis with confidence score.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TempoHypothesis<T>
{
    /// <summary>
    /// Gets or sets the tempo in BPM.
    /// </summary>
    public T Tempo { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence score (0.0 to 1.0).
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets whether this is a half-tempo or double-tempo variant.
    /// </summary>
    public TempoRelation Relation { get; set; } = TempoRelation.Primary;
}

/// <summary>
/// Relationship of a tempo hypothesis to the primary tempo.
/// </summary>
public enum TempoRelation
{
    /// <summary>Primary tempo estimate.</summary>
    Primary,
    /// <summary>Half of the primary tempo.</summary>
    HalfTime,
    /// <summary>Double the primary tempo.</summary>
    DoubleTime,
    /// <summary>Alternative independent estimate.</summary>
    Alternative
}

/// <summary>
/// Result of downbeat detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DownbeatResult<T>
{
    /// <summary>
    /// Gets or sets the downbeat positions in seconds.
    /// </summary>
    public IReadOnlyList<double> DownbeatTimes { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets the detected time signature.
    /// </summary>
    public TimeSignature TimeSignature { get; set; } = new TimeSignature(4, 4);

    /// <summary>
    /// Gets or sets the measure boundaries in seconds.
    /// </summary>
    public IReadOnlyList<double> MeasureStarts { get; set; } = Array.Empty<double>();
}

/// <summary>
/// Represents a musical time signature.
/// </summary>
public class TimeSignature
{
    /// <summary>
    /// Gets or sets the numerator (beats per measure).
    /// </summary>
    public int Numerator { get; set; }

    /// <summary>
    /// Gets or sets the denominator (beat unit).
    /// </summary>
    public int Denominator { get; set; }

    /// <summary>
    /// Initializes a new time signature.
    /// </summary>
    public TimeSignature(int numerator, int denominator)
    {
        Numerator = numerator;
        Denominator = denominator;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Numerator}/{Denominator}";
}
