using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for speech emotion recognition models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speech Emotion Recognition (SER) identifies emotional states from voice:
/// <list type="bullet">
/// <item><description>Basic emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral</description></item>
/// <item><description>Arousal: Low (calm) to High (excited)</description></item>
/// <item><description>Valence: Negative to Positive</description></item>
/// <item><description>Dominance: Submissive to Dominant</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> This is like reading emotions from someone's voice!
///
/// How humans convey emotion in speech:
/// - Pitch: Higher when excited/happy, lower when sad
/// - Speed: Faster when angry/excited, slower when sad
/// - Volume: Louder when angry, softer when sad/fearful
/// - Voice quality: Breathy, tense, relaxed
///
/// Applications:
/// - Call centers: Detect frustrated customers for escalation
/// - Mental health: Monitor patient emotional state
/// - Voice assistants: Respond appropriately to user mood
/// - Gaming: Adapt game difficulty/story based on player emotion
/// - Market research: Analyze focus group reactions
///
/// Challenges:
/// - Cultural differences in emotional expression
/// - Speaker variability (age, gender, accent)
/// - Context dependency (same words can mean different emotions)
/// - Mixed emotions (happy but nervous)
/// </para>
/// </remarks>
public interface IEmotionRecognizer<T>
{
    /// <summary>
    /// Gets the sample rate this recognizer operates at.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of emotions this model can detect.
    /// </summary>
    IReadOnlyList<string> SupportedEmotions { get; }

    /// <summary>
    /// Recognizes the primary emotion in speech audio.
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <returns>The detected emotion and confidence score.</returns>
    EmotionResult<T> RecognizeEmotion(Tensor<T> audio);

    /// <summary>
    /// Gets probabilities for all supported emotions.
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <returns>Dictionary mapping emotion names to probabilities.</returns>
    IReadOnlyDictionary<string, T> GetEmotionProbabilities(Tensor<T> audio);

    /// <summary>
    /// Recognizes emotions over time (for longer recordings).
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <param name="windowSizeMs">Analysis window size in milliseconds.</param>
    /// <param name="hopSizeMs">Hop between windows in milliseconds.</param>
    /// <returns>Time-series of emotion predictions.</returns>
    IReadOnlyList<TimedEmotionResult<T>> RecognizeEmotionTimeSeries(
        Tensor<T> audio,
        int windowSizeMs = 1000,
        int hopSizeMs = 500);

    /// <summary>
    /// Gets arousal (activation) level from speech.
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <returns>Arousal level from -1.0 (calm) to 1.0 (excited).</returns>
    T GetArousal(Tensor<T> audio);

    /// <summary>
    /// Gets valence (positivity) level from speech.
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <returns>Valence level from -1.0 (negative) to 1.0 (positive).</returns>
    T GetValence(Tensor<T> audio);

    /// <summary>
    /// Extracts emotion-relevant features from audio.
    /// </summary>
    /// <param name="audio">Audio tensor.</param>
    /// <returns>Feature vector useful for emotion classification.</returns>
    Vector<T> ExtractEmotionFeatures(Tensor<T> audio);
}

/// <summary>
/// Represents the result of emotion recognition.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class EmotionResult<T>
{
    /// <summary>
    /// Gets the primary detected emotion.
    /// </summary>
    public required string Emotion { get; init; }

    /// <summary>
    /// Gets the confidence score (0.0 to 1.0).
    /// </summary>
    public required T Confidence { get; init; }

    /// <summary>
    /// Gets the secondary emotion (if detected with significant confidence).
    /// </summary>
    public string? SecondaryEmotion { get; init; }

    /// <summary>
    /// Gets the arousal level (-1.0 to 1.0).
    /// </summary>
    public T? Arousal { get; init; }

    /// <summary>
    /// Gets the valence level (-1.0 to 1.0).
    /// </summary>
    public T? Valence { get; init; }
}

/// <summary>
/// Represents a timed emotion prediction.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class TimedEmotionResult<T> : EmotionResult<T>
{
    /// <summary>
    /// Gets the start time in seconds.
    /// </summary>
    public required double StartTime { get; init; }

    /// <summary>
    /// Gets the end time in seconds.
    /// </summary>
    public required double EndTime { get; init; }
}
