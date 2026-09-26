using AiDotNet.NeuralNetworks;

namespace AiDotNet.Audio.LanguageIdentification;

/// <summary>
/// The label set and display names shared by the language identifiers that do not carry their own.
/// </summary>
/// <remarks>
/// ECAPATDNNLanguageIdentifier and Wav2Vec2LanguageIdentifier each kept an identical private copy of both
/// tables, so a change to one would silently diverge from the other. VoxLingua107Identifier keeps its own
/// 107-language set because that set is the dataset's label set.
/// </remarks>
internal static class LanguageIdentificationDefaults
{
    /// <summary>
    /// The default label set, used when an identifier is loaded without a language list (the ONNX path).
    /// </summary>
    internal static readonly IReadOnlyList<string> CommonLanguageCodes = new[]
    {
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
        "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "cs"
    };

    private static readonly Dictionary<string, string> DisplayNames = new Dictionary<string, string>
    {
        ["en"] = "English",
        ["es"] = "Spanish",
        ["fr"] = "French",
        ["de"] = "German",
        ["it"] = "Italian",
        ["pt"] = "Portuguese",
        ["ru"] = "Russian",
        ["zh"] = "Chinese",
        ["ja"] = "Japanese",
        ["ko"] = "Korean",
        ["ar"] = "Arabic",
        ["hi"] = "Hindi",
        ["tr"] = "Turkish",
        ["pl"] = "Polish",
        ["nl"] = "Dutch",
        ["sv"] = "Swedish",
        ["da"] = "Danish",
        ["no"] = "Norwegian",
        ["fi"] = "Finnish",
        ["cs"] = "Czech",
        ["el"] = "Greek",
        ["he"] = "Hebrew",
        ["th"] = "Thai",
        ["vi"] = "Vietnamese",
        ["id"] = "Indonesian",
        ["ms"] = "Malay",
        ["uk"] = "Ukrainian",
        ["ro"] = "Romanian",
        ["hu"] = "Hungarian",
        ["bg"] = "Bulgarian"
    };

    /// <summary>
    /// Returns a fresh code-to-name map, so no identifier can mutate another's.
    /// </summary>
    internal static Dictionary<string, string> CreateDisplayNameMap()
        => new Dictionary<string, string>(DisplayNames);

    /// <summary>
    /// Rejects an architecture whose declared output width contradicts the language list that sizes the
    /// classifier head.
    /// </summary>
    /// <remarks>
    /// An output size of 0 (the architecture default) leaves the decision to the language list.
    /// </remarks>
    internal static void ValidateHeadWidth<T>(NeuralNetworkArchitecture<T> architecture, int languageCount)
    {
        if (architecture.OutputSize > 0 && architecture.OutputSize != languageCount)
        {
            throw new ArgumentException(
                $"The architecture declares an output size of {architecture.OutputSize}, but the classifier " +
                $"head is sized by the {languageCount} supported languages. Set the output size to " +
                $"{languageCount}, or leave it at 0 so the language list decides.",
                nameof(architecture));
        }
    }
}
