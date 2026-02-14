namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for audio event detection models that identify specific sounds/events in audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio event detection identifies when specific sounds occur in an audio stream.
/// Unlike classification which assigns one label to entire clips, event detection
/// finds multiple events with their timestamps.
/// </para>
/// <para>
/// <b>For Beginners:</b> Event detection is like having a listener who notes down
/// every distinct sound they hear and when it happened.
///
/// How it works:
/// 1. Audio is analyzed in overlapping windows
/// 2. Each window is classified for the presence of various events
/// 3. Consecutive detections are merged into event segments
///
/// Types of events:
/// - Environmental: Car horn, dog bark, siren, glass breaking
/// - Speech: Laughter, cough, scream, applause
/// - Music: Drum hit, guitar strum, piano note
/// - Industrial: Machine alarm, tool sounds
///
/// Use cases:
/// - Security/surveillance (detect gunshots, breaking glass)
/// - Smart home (detect doorbell, smoke alarm, baby crying)
/// - Wildlife monitoring (detect animal calls)
/// - Content moderation (detect inappropriate sounds)
/// - Accessibility (alert deaf users to sounds)
///
/// Challenges:
/// - Overlapping events (multiple sounds at once)
/// - Variable event duration (short beep vs long siren)
/// - Background noise interference
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioEventDetector")]
public interface IAudioEventDetector<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of event types this model can detect.
    /// </summary>
    IReadOnlyList<string> SupportedEvents { get; }

    /// <summary>
    /// Gets the time resolution for event detection in seconds.
    /// </summary>
    double TimeResolution { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Detects audio events in the audio stream.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <param name="threshold">Detection threshold (0.0 to 1.0). Lower = more sensitive.</param>
    /// <returns>Event detection result with detected events.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for detecting events.
    /// - Pass in audio
    /// - Get back a list of detected sounds and when they occurred
    /// </para>
    /// </remarks>
    AudioEventResult<T> Detect(Tensor<T> audio);

    /// <summary>
    /// Detects audio events in the audio stream with custom threshold.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <param name="threshold">Detection threshold (0.0 to 1.0). Lower = more sensitive.</param>
    /// <returns>Event detection result with detected events.</returns>
    AudioEventResult<T> Detect(Tensor<T> audio, T threshold);

    /// <summary>
    /// Detects audio events asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Event detection result.</returns>
    Task<AudioEventResult<T>> DetectAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Detects specific events only.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="eventTypes">Event types to detect.</param>
    /// <returns>Event detection result filtered to specified types.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you only care about specific sounds.
    /// - DetectSpecific(audio, ["dog_bark", "siren"]) only looks for dogs and sirens
    /// </para>
    /// </remarks>
    AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes);

    /// <summary>
    /// Detects specific events only with custom threshold.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="eventTypes">Event types to detect.</param>
    /// <param name="threshold">Detection threshold.</param>
    /// <returns>Event detection result filtered to specified types.</returns>
    AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold);

    /// <summary>
    /// Gets frame-level event probabilities.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tensor of event probabilities [time_frames, num_events].</returns>
    /// <remarks>
    /// <para>
    /// Useful for visualization or custom post-processing of detections.
    /// </para>
    /// </remarks>
    Tensor<T> GetEventProbabilities(Tensor<T> audio);

    /// <summary>
    /// Performs real-time event detection on a streaming session.
    /// </summary>
    /// <returns>Streaming session for real-time detection.</returns>
    IStreamingEventDetectionSession<T> StartStreamingSession();

    /// <summary>
    /// Performs real-time event detection on a streaming session with custom settings.
    /// </summary>
    /// <param name="sampleRate">Sample rate of incoming audio.</param>
    /// <param name="threshold">Detection threshold.</param>
    /// <returns>Streaming session for real-time detection.</returns>
    IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold);
}

/// <summary>
/// Result of audio event detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AudioEventResult<T> : IEnumerable<AudioEvent<T>>
{
    /// <summary>
    /// Gets or sets the detected events.
    /// </summary>
    public IReadOnlyList<AudioEvent<T>> Events { get; set; } = Array.Empty<AudioEvent<T>>();

    /// <summary>
    /// Gets or sets the total duration in seconds.
    /// </summary>
    public double TotalDuration { get; set; }

    /// <summary>
    /// Gets or sets the unique event types detected.
    /// </summary>
    public IReadOnlyList<string> DetectedEventTypes { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets event statistics.
    /// </summary>
    public IReadOnlyDictionary<string, EventStatistics<T>> EventStats { get; set; } =
        new Dictionary<string, EventStatistics<T>>();

    /// <summary>
    /// Gets events of a specific type.
    /// </summary>
    /// <param name="eventType">The event type to filter for.</param>
    /// <returns>Events of the specified type.</returns>
    public IEnumerable<AudioEvent<T>> GetEventsByType(string eventType) =>
        Events.Where(e => e.EventType == eventType);

    /// <summary>
    /// Returns an enumerator that iterates through the events.
    /// </summary>
    public IEnumerator<AudioEvent<T>> GetEnumerator() => Events.GetEnumerator();

    /// <summary>
    /// Returns an enumerator that iterates through the events.
    /// </summary>
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => Events.GetEnumerator();
}

/// <summary>
/// A detected audio event.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AudioEvent<T>
{
    /// <summary>
    /// Gets or sets the event type/label.
    /// </summary>
    public string EventType { get; set; } = string.Empty;

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

    /// <summary>
    /// Gets or sets the peak confidence time within the event.
    /// </summary>
    public double PeakTime { get; set; }

    /// <summary>
    /// Gets the duration of the event in seconds.
    /// </summary>
    public double Duration => EndTime - StartTime;
}

/// <summary>
/// Statistics for an event type.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EventStatistics<T>
{
    /// <summary>
    /// Gets or sets the number of occurrences.
    /// </summary>
    public int Count { get; set; }

    /// <summary>
    /// Gets or sets the total duration of all occurrences.
    /// </summary>
    public double TotalDuration { get; set; }

    /// <summary>
    /// Gets or sets the average confidence across occurrences.
    /// </summary>
    public T AverageConfidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the maximum confidence.
    /// </summary>
    public T MaxConfidence { get; set; } = default!;
}

/// <summary>
/// Interface for streaming event detection sessions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IStreamingEventDetectionSession<T> : IDisposable
{
    /// <summary>
    /// Feeds audio samples to the detection session.
    /// </summary>
    /// <param name="audioChunk">Audio samples to process.</param>
    void FeedAudio(Tensor<T> audioChunk);

    /// <summary>
    /// Gets newly detected events since last call.
    /// </summary>
    /// <returns>Newly detected events.</returns>
    IReadOnlyList<AudioEvent<T>> GetNewEvents();

    /// <summary>
    /// Gets current detection state for all event types.
    /// </summary>
    /// <returns>Current probabilities for each event type.</returns>
    IReadOnlyDictionary<string, T> GetCurrentState();

    /// <summary>
    /// Event raised when a new event is detected.
    /// </summary>
    event EventHandler<AudioEvent<T>>? EventDetected;
}
