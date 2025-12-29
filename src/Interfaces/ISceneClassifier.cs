namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for acoustic scene classification models that identify the environment/context of audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Acoustic scene classification (ASC) identifies the environment or context where audio was recorded.
/// Unlike event detection which finds specific sounds, scene classification characterizes the overall
/// acoustic atmosphere.
/// </para>
/// <para>
/// <b>For Beginners:</b> Scene classification is like asking "Where was this recording made?"
///
/// How it works:
/// 1. Audio features capture the overall acoustic character
/// 2. A classifier matches these features to known scene types
/// 3. The most likely scene (and alternatives) are returned
///
/// Example scenes:
/// - Indoor: Office, restaurant, kitchen, library, shopping mall
/// - Outdoor: Park, street, beach, forest, construction site
/// - Transportation: Car, bus, train, metro, airport
///
/// How scenes differ from events:
/// - Event: "A dog barked" (specific sound)
/// - Scene: "This was recorded in a park" (overall environment)
///
/// Use cases:
/// - Context-aware devices (adjust phone behavior based on location)
/// - Audio organization (group recordings by location)
/// - Surveillance (detect unusual environments)
/// - AR/VR (match virtual audio to real environment)
/// - Assistive technology (describe environment to blind users)
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
public interface ISceneClassifier<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of scenes this model can classify.
    /// </summary>
    IReadOnlyList<string> SupportedScenes { get; }

    /// <summary>
    /// Gets the minimum audio duration required for reliable classification.
    /// </summary>
    double MinimumDurationSeconds { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Classifies the acoustic scene of audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <returns>Scene classification result.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for identifying the scene.
    /// - Pass in a recording
    /// - Get back where it was likely recorded (office, park, etc.)
    /// </para>
    /// </remarks>
    SceneClassificationResult<T> Classify(Tensor<T> audio);

    /// <summary>
    /// Classifies acoustic scene asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Scene classification result.</returns>
    Task<SceneClassificationResult<T>> ClassifyAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets scene probabilities for all supported scenes.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Dictionary mapping scene names to probability scores.</returns>
    IReadOnlyDictionary<string, T> GetSceneProbabilities(Tensor<T> audio);

    /// <summary>
    /// Gets top-K scene predictions.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="k">Number of top scenes to return.</param>
    /// <returns>List of top scene predictions.</returns>
    IReadOnlyList<ScenePrediction<T>> GetTopScenes(Tensor<T> audio, int k = 5);

    /// <summary>
    /// Tracks scene changes over time in longer audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="segmentDuration">Duration of each analysis segment in seconds.</param>
    /// <returns>Scene tracking result showing scene over time.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For longer recordings that might move between places
    /// (like walking from street to inside a building), this tracks the scene changes.
    /// </para>
    /// </remarks>
    SceneTrackingResult<T> TrackSceneChanges(Tensor<T> audio, double segmentDuration = 10.0);

    /// <summary>
    /// Extracts acoustic features used for scene classification.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Feature tensor capturing acoustic characteristics.</returns>
    Tensor<T> ExtractAcousticFeatures(Tensor<T> audio);
}

/// <summary>
/// Result of scene classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SceneClassificationResult<T>
{
    /// <summary>
    /// Gets or sets the predicted scene.
    /// </summary>
    public string PredictedScene { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the scene category (indoor/outdoor/transportation).
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets all scene predictions sorted by probability.
    /// </summary>
    public IReadOnlyList<ScenePrediction<T>> AllScenes { get; set; } = Array.Empty<ScenePrediction<T>>();

    /// <summary>
    /// Gets or sets detected acoustic characteristics.
    /// </summary>
    public AcousticCharacteristics<T>? Characteristics { get; set; }
}

/// <summary>
/// A single scene prediction with confidence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ScenePrediction<T>
{
    /// <summary>
    /// Gets or sets the scene name.
    /// </summary>
    public string Scene { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the scene category.
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the probability/confidence score.
    /// </summary>
    public T Probability { get; set; } = default!;

    /// <summary>
    /// Gets or sets the rank (1 = most likely).
    /// </summary>
    public int Rank { get; set; }
}

/// <summary>
/// Acoustic characteristics of a scene.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AcousticCharacteristics<T>
{
    /// <summary>
    /// Gets or sets the estimated reverberation level.
    /// </summary>
    public T ReverberationLevel { get; set; } = default!;

    /// <summary>
    /// Gets or sets the estimated background noise level.
    /// </summary>
    public T BackgroundNoiseLevel { get; set; } = default!;

    /// <summary>
    /// Gets or sets whether the environment appears to be indoor.
    /// </summary>
    public bool IsIndoor { get; set; }

    /// <summary>
    /// Gets or sets the estimated crowd density (low/medium/high/none).
    /// </summary>
    public string CrowdDensity { get; set; } = "none";

    /// <summary>
    /// Gets or sets whether traffic sounds are present.
    /// </summary>
    public bool HasTrafficSounds { get; set; }

    /// <summary>
    /// Gets or sets whether nature sounds are present.
    /// </summary>
    public bool HasNatureSounds { get; set; }
}

/// <summary>
/// Result of tracking scene changes over time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SceneTrackingResult<T>
{
    /// <summary>
    /// Gets or sets the overall dominant scene.
    /// </summary>
    public string DominantScene { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets scene segments over time.
    /// </summary>
    public IReadOnlyList<SceneSegment<T>> Segments { get; set; } = Array.Empty<SceneSegment<T>>();

    /// <summary>
    /// Gets or sets detected scene transitions.
    /// </summary>
    public IReadOnlyList<SceneTransition<T>> Transitions { get; set; } = Array.Empty<SceneTransition<T>>();

    /// <summary>
    /// Gets or sets whether scene changes were detected.
    /// </summary>
    public bool HasSceneChanges { get; set; }

    /// <summary>
    /// Gets or sets scene distribution over the recording.
    /// </summary>
    public IReadOnlyDictionary<string, double> SceneDistribution { get; set; } = new Dictionary<string, double>();
}

/// <summary>
/// A segment with scene information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SceneSegment<T>
{
    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets or sets the scene in this segment.
    /// </summary>
    public string Scene { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;
}

/// <summary>
/// A detected scene transition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SceneTransition<T>
{
    /// <summary>
    /// Gets or sets the time of transition in seconds.
    /// </summary>
    public double Time { get; set; }

    /// <summary>
    /// Gets or sets the scene before transition.
    /// </summary>
    public string FromScene { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the scene after transition.
    /// </summary>
    public string ToScene { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence in the transition detection.
    /// </summary>
    public T Confidence { get; set; } = default!;
}
