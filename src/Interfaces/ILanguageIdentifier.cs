using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for spoken language identification from audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Language Identification (LID) determines which language is being spoken
/// in an audio recording. This is different from speech recognition - we're
/// identifying the language, not transcribing the words.
/// </para>
/// <para><b>For Beginners:</b> This is like having a friend who can tell you
/// "that's French!" or "that sounds like Mandarin!" just from hearing it.
///
/// How it works:
/// 1. Extract acoustic features (phonemes, prosody, rhythm)
/// 2. Compare to language models trained on many languages
/// 3. Return the most likely language(s)
///
/// Applications:
/// - Call routing in multilingual call centers
/// - Automatic subtitle language selection
/// - Content moderation (filter by language)
/// - Multilingual speech recognition (select correct model)
/// - Immigration/border control voice analysis
///
/// Challenges:
/// - Code-switching (mixing languages mid-sentence)
/// - Accented speech (Spanish with American accent)
/// - Closely related languages (Norwegian vs Swedish)
/// - Short utterances (harder to identify with less audio)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("LanguageIdentifier")]
public interface ILanguageIdentifier<T>
{
    /// <summary>
    /// Gets the sample rate this identifier operates at.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the list of languages this model can identify.
    /// </summary>
    /// <remarks>
    /// Language codes typically follow ISO 639-1 (e.g., "en", "es", "zh")
    /// or ISO 639-3 for more specific variants.
    /// </remarks>
    IReadOnlyList<string> SupportedLanguages { get; }

    /// <summary>
    /// Identifies the language spoken in audio.
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <returns>Detected language code and confidence.</returns>
    LanguageResult<T> IdentifyLanguage(Tensor<T> audio);

    /// <summary>
    /// Gets probabilities for all supported languages.
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <returns>Dictionary mapping language codes to probabilities.</returns>
    IReadOnlyDictionary<string, T> GetLanguageProbabilities(Tensor<T> audio);

    /// <summary>
    /// Gets the top-N most likely languages.
    /// </summary>
    /// <param name="audio">Audio tensor containing speech.</param>
    /// <param name="topN">Number of languages to return.</param>
    /// <returns>List of (language, probability) pairs sorted by probability.</returns>
    IReadOnlyList<(string Language, T Probability)> GetTopLanguages(Tensor<T> audio, int topN = 5);

    /// <summary>
    /// Identifies language with time segmentation (for multilingual audio).
    /// </summary>
    /// <param name="audio">Audio tensor that may contain multiple languages.</param>
    /// <param name="windowSizeMs">Analysis window size in milliseconds.</param>
    /// <returns>Time-segmented language predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this when someone might switch languages
    /// mid-recording (code-switching). It tells you which language is spoken
    /// at each point in time.
    /// </para>
    /// </remarks>
    IReadOnlyList<LanguageSegment<T>> IdentifyLanguageSegments(
        Tensor<T> audio,
        int windowSizeMs = 2000);

    /// <summary>
    /// Gets the display name for a language code.
    /// </summary>
    /// <param name="languageCode">ISO language code.</param>
    /// <returns>Human-readable language name.</returns>
    string GetLanguageDisplayName(string languageCode);

    /// <summary>
    /// Checks if two audio samples are in the same language.
    /// </summary>
    /// <param name="audio1">First audio sample.</param>
    /// <param name="audio2">Second audio sample.</param>
    /// <returns>True if same language, with confidence score.</returns>
    (bool SameLanguage, T Confidence) AreSameLanguage(Tensor<T> audio1, Tensor<T> audio2);
}

/// <summary>
/// Represents the result of language identification.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class LanguageResult<T>
{
    /// <summary>
    /// Gets the detected language code (ISO 639-1/3).
    /// </summary>
    public required string LanguageCode { get; init; }

    /// <summary>
    /// Gets the human-readable language name.
    /// </summary>
    public required string LanguageName { get; init; }

    /// <summary>
    /// Gets the confidence score (0.0 to 1.0).
    /// </summary>
    public required T Confidence { get; init; }

    /// <summary>
    /// Gets the second most likely language (if close in probability).
    /// </summary>
    public string? AlternativeLanguage { get; init; }

    /// <summary>
    /// Gets the probability of the alternative language.
    /// </summary>
    public T? AlternativeProbability { get; init; }
}

/// <summary>
/// Represents a time-segmented language detection result.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class LanguageSegment<T>
{
    /// <summary>
    /// Gets the start time in seconds.
    /// </summary>
    public required double StartTime { get; init; }

    /// <summary>
    /// Gets the end time in seconds.
    /// </summary>
    public required double EndTime { get; init; }

    /// <summary>
    /// Gets the detected language code.
    /// </summary>
    public required string LanguageCode { get; init; }

    /// <summary>
    /// Gets the confidence score.
    /// </summary>
    public required T Confidence { get; init; }
}
