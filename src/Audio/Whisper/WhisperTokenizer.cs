namespace AiDotNet.Audio.Whisper;

/// <summary>
/// Tokenizer for Whisper speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// Whisper uses a special tokenizer with BPE (Byte Pair Encoding) and special tokens
/// for controlling transcription behavior (language, task, timestamps).
/// </para>
/// <para><b>For Beginners:</b> A tokenizer converts text to numbers (tokens) and back.
/// Whisper's tokenizer has special tokens for:
/// <list type="bullet">
/// <item>Language codes (to specify which language to transcribe)</item>
/// <item>Task tokens (transcribe vs translate)</item>
/// <item>Timestamp tokens (for word-level timing)</item>
/// </list>
/// </para>
/// </remarks>
public class WhisperTokenizer
{
    // Standard Whisper special token IDs
    private const int EndOfTextId = 50256;
    private const int StartOfTranscriptId = 50257;
    private const int TranslateTokenId = 50357;
    private const int TranscribeTokenId = 50358;
    private const int NoSpeechTokenId = 50361;
    private const int NoTimestampsTokenId = 50362;
    private const int TimestampBeginId = 50363;

    // Language token mapping (Whisper multilingual)
    private static readonly Dictionary<string, int> LanguageTokens = new()
    {
        ["en"] = 50259, // English
        ["zh"] = 50260, // Chinese
        ["de"] = 50261, // German
        ["es"] = 50262, // Spanish
        ["ru"] = 50263, // Russian
        ["ko"] = 50264, // Korean
        ["fr"] = 50265, // French
        ["ja"] = 50266, // Japanese
        ["pt"] = 50267, // Portuguese
        ["tr"] = 50268, // Turkish
        ["pl"] = 50269, // Polish
        ["ca"] = 50270, // Catalan
        ["nl"] = 50271, // Dutch
        ["ar"] = 50272, // Arabic
        ["sv"] = 50273, // Swedish
        ["it"] = 50274, // Italian
        ["id"] = 50275, // Indonesian
        ["hi"] = 50276, // Hindi
        ["fi"] = 50277, // Finnish
        ["vi"] = 50278, // Vietnamese
        ["he"] = 50279, // Hebrew
        ["uk"] = 50280, // Ukrainian
        ["el"] = 50281, // Greek
        ["ms"] = 50282, // Malay
        ["cs"] = 50283, // Czech
        ["ro"] = 50284, // Romanian
        ["da"] = 50285, // Danish
        ["hu"] = 50286, // Hungarian
        ["ta"] = 50287, // Tamil
        ["no"] = 50288, // Norwegian
        ["th"] = 50289, // Thai
        ["ur"] = 50290, // Urdu
        ["hr"] = 50291, // Croatian
        ["bg"] = 50292, // Bulgarian
        ["lt"] = 50293, // Lithuanian
        ["la"] = 50294, // Latin
        ["mi"] = 50295, // Maori
        ["ml"] = 50296, // Malayalam
        ["cy"] = 50297, // Welsh
        ["sk"] = 50298, // Slovak
        ["te"] = 50299, // Telugu
        ["fa"] = 50300, // Persian
        ["lv"] = 50301, // Latvian
        ["bn"] = 50302, // Bengali
        ["sr"] = 50303, // Serbian
        ["az"] = 50304, // Azerbaijani
        ["sl"] = 50305, // Slovenian
        ["kn"] = 50306, // Kannada
        ["et"] = 50307, // Estonian
        ["mk"] = 50308, // Macedonian
        ["br"] = 50309, // Breton
        ["eu"] = 50310, // Basque
        ["is"] = 50311, // Icelandic
        ["hy"] = 50312, // Armenian
        ["ne"] = 50313, // Nepali
        ["mn"] = 50314, // Mongolian
        ["bs"] = 50315, // Bosnian
        ["kk"] = 50316, // Kazakh
        ["sq"] = 50317, // Albanian
        ["sw"] = 50318, // Swahili
        ["gl"] = 50319, // Galician
        ["mr"] = 50320, // Marathi
        ["pa"] = 50321, // Punjabi
        ["si"] = 50322, // Sinhala
        ["km"] = 50323, // Khmer
        ["sn"] = 50324, // Shona
        ["yo"] = 50325, // Yoruba
        ["so"] = 50326, // Somali
        ["af"] = 50327, // Afrikaans
        ["oc"] = 50328, // Occitan
        ["ka"] = 50329, // Georgian
        ["be"] = 50330, // Belarusian
        ["tg"] = 50331, // Tajik
        ["sd"] = 50332, // Sindhi
        ["gu"] = 50333, // Gujarati
        ["am"] = 50334, // Amharic
        ["yi"] = 50335, // Yiddish
        ["lo"] = 50336, // Lao
        ["uz"] = 50337, // Uzbek
        ["fo"] = 50338, // Faroese
        ["ht"] = 50339, // Haitian Creole
        ["ps"] = 50340, // Pashto
        ["tk"] = 50341, // Turkmen
        ["nn"] = 50342, // Norwegian Nynorsk
        ["mt"] = 50343, // Maltese
        ["sa"] = 50344, // Sanskrit
        ["lb"] = 50345, // Luxembourgish
        ["my"] = 50346, // Myanmar
        ["bo"] = 50347, // Tibetan
        ["tl"] = 50348, // Tagalog
        ["mg"] = 50349, // Malagasy
        ["as"] = 50350, // Assamese
        ["tt"] = 50351, // Tatar
        ["haw"] = 50352, // Hawaiian
        ["ln"] = 50353, // Lingala
        ["ha"] = 50354, // Hausa
        ["ba"] = 50355, // Bashkir
        ["jw"] = 50356, // Javanese
    };

    /// <summary>
    /// Gets the end of text token ID.
    /// </summary>
    public int EndOfText => EndOfTextId;

    /// <summary>
    /// Gets the start of transcript token ID.
    /// </summary>
    public int StartOfTranscript => StartOfTranscriptId;

    /// <summary>
    /// Gets the translate task token ID.
    /// </summary>
    public int TranslateToken => TranslateTokenId;

    /// <summary>
    /// Gets the transcribe task token ID.
    /// </summary>
    public int TranscribeToken => TranscribeTokenId;

    /// <summary>
    /// Gets the no speech token ID.
    /// </summary>
    public int NoSpeechToken => NoSpeechTokenId;

    /// <summary>
    /// Gets the no timestamps token ID.
    /// </summary>
    public int NoTimestampsToken => NoTimestampsTokenId;

    /// <summary>
    /// Gets all supported language codes.
    /// </summary>
    public static IReadOnlyList<string> SupportedLanguages => LanguageTokens.Keys.ToList();

    /// <summary>
    /// Gets the token ID for a language code.
    /// </summary>
    /// <param name="languageCode">Two-letter language code (e.g., "en", "es").</param>
    /// <returns>The token ID for the language.</returns>
    public int GetLanguageToken(string languageCode)
    {
        if (LanguageTokens.TryGetValue(languageCode.ToLowerInvariant(), out int tokenId))
        {
            return tokenId;
        }

        throw new ArgumentException($"Unsupported language code: {languageCode}", nameof(languageCode));
    }

    /// <summary>
    /// Gets the timestamp token ID for a given time in seconds.
    /// </summary>
    /// <param name="timeSeconds">Time in seconds (must be a multiple of 0.02).</param>
    /// <returns>The timestamp token ID.</returns>
    public int GetTimestampToken(double timeSeconds)
    {
        // Whisper uses 20ms precision for timestamps
        int index = (int)(timeSeconds / 0.02);
        return TimestampBeginId + index;
    }

    /// <summary>
    /// Converts a timestamp token ID to time in seconds.
    /// </summary>
    /// <param name="tokenId">The timestamp token ID.</param>
    /// <returns>Time in seconds.</returns>
    public double GetTimeFromToken(int tokenId)
    {
        if (tokenId < TimestampBeginId)
        {
            throw new ArgumentException("Token is not a timestamp token.", nameof(tokenId));
        }

        int index = tokenId - TimestampBeginId;
        return index * 0.02;
    }

    /// <summary>
    /// Checks if a token ID is a special token.
    /// </summary>
    /// <param name="tokenId">The token ID to check.</param>
    /// <returns>True if the token is a special token.</returns>
    public bool IsSpecialToken(int tokenId)
    {
        return tokenId >= EndOfTextId;
    }

    /// <summary>
    /// Checks if a token ID is a timestamp token.
    /// </summary>
    /// <param name="tokenId">The token ID to check.</param>
    /// <returns>True if the token is a timestamp token.</returns>
    public bool IsTimestampToken(int tokenId)
    {
        return tokenId >= TimestampBeginId;
    }

    /// <summary>
    /// Decodes a sequence of token IDs to text.
    /// </summary>
    /// <param name="tokenIds">The token IDs to decode.</param>
    /// <returns>The decoded text.</returns>
    /// <remarks>
    /// This is a simplified decoder. A full implementation would use
    /// the actual BPE vocabulary from the Whisper model.
    /// </remarks>
    public string Decode(IEnumerable<long> tokenIds)
    {
        var tokens = tokenIds.Where(t => !IsSpecialToken((int)t)).ToList();

        // Note: This is a placeholder implementation.
        // A full implementation would use the BPE vocabulary to decode.
        // For now, we return a representation of the tokens.
        if (tokens.Count == 0)
        {
            return string.Empty;
        }

        // In a full implementation, each token would be looked up in the vocabulary
        // and converted to its text representation.
        return $"[Decoded {tokens.Count} tokens - full BPE decoder not implemented]";
    }

    /// <summary>
    /// Encodes text to token IDs.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>The encoded token IDs.</returns>
    /// <remarks>
    /// This is a placeholder. A full implementation would use BPE encoding.
    /// </remarks>
    public List<long> Encode(string text)
    {
        // Placeholder - a full implementation would use BPE encoding
        // with the Whisper vocabulary
        return [];
    }
}
