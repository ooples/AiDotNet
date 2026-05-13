using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WhisperTokenizer
{
    // GPT-2 / Whisper byte-level BPE state — populated lazily by LoadVocab.
    // Null means "vocab not loaded; use byte-level identity fallback".
    private Dictionary<string, int>? _vocab;          // token-string → ID
    private Dictionary<int, string>? _inverseVocab;   // ID → token-string
    private Dictionary<(string, string), int>? _merges; // pair → priority (lower = applied earlier)
    private readonly Dictionary<int, char> _byteToUnicode;
    private readonly Dictionary<char, int> _unicodeToByte;
    // GPT-2's pre-tokenizer regex (Radford 2019, OpenAI tiktoken's gpt2 pattern).
    // Splits on contractions, alphabetic/numeric/punctuation runs, leading spaces.
    private static readonly Regex _preTokenPattern = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    /// <summary>Initializes a tokenizer that operates in byte-level identity mode (no learned merges).</summary>
    public WhisperTokenizer()
    {
        (_byteToUnicode, _unicodeToByte) = BuildByteUnicodeMaps();
    }

    /// <summary>Initializes a tokenizer that loads the official Whisper / GPT-2 BPE vocabulary.</summary>
    /// <param name="vocabPath">Path to vocab.json (token string → ID).</param>
    /// <param name="mergesPath">Path to merges.txt (priority-ordered pair merges).</param>
    /// <remarks>
    /// Internal — asset initialisation is owned by the WhisperModel / model
    /// builder layer; users go through that facade rather than instantiating
    /// the tokenizer directly.
    /// </remarks>
    internal WhisperTokenizer(string vocabPath, string mergesPath) : this()
    {
        LoadVocab(vocabPath, mergesPath);
    }

    /// <summary>Loads the BPE vocabulary and merge table at runtime.</summary>
    /// <remarks>
    /// vocab.json maps every byte-unicode token to its integer ID; merges.txt
    /// lists the learned pair merges in priority order (first line = priority
    /// 0 = applied first). Both files are shipped alongside any HuggingFace
    /// Whisper checkpoint. Internal — see the file-path constructor.
    /// </remarks>
    internal void LoadVocab(string vocabPath, string mergesPath)
    {
        if (vocabPath is null) throw new ArgumentNullException(nameof(vocabPath));
        if (mergesPath is null) throw new ArgumentNullException(nameof(mergesPath));
        if (!File.Exists(vocabPath)) throw new FileNotFoundException("vocab.json not found.", vocabPath);
        if (!File.Exists(mergesPath)) throw new FileNotFoundException("merges.txt not found.", mergesPath);

        // vocab.json: {"token": id, ...} where keys are the byte-unicode token strings.
        var vocabJson = File.ReadAllText(vocabPath);
        var vocab = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson)
            ?? throw new InvalidDataException("vocab.json could not be parsed as a string→int dictionary.");

        // merges.txt: optional "#version: ..." header line then pair-per-line.
        // Reject malformed rows with a precise line-number diagnostic. The
        // previous `continue` swallowed bad rows and built a partial merge
        // table — BPE tokenization would then run silently with corrupted
        // priority ranks, producing token streams subtly different from
        // the reference Whisper tokenizer.
        var merges = new Dictionary<(string, string), int>();
        int priority = 0;
        int lineNumber = 0;
        foreach (var rawLine in File.ReadLines(mergesPath))
        {
            lineNumber++;
            var line = rawLine.TrimEnd();
            if (string.IsNullOrEmpty(line)) continue;
            if (line.StartsWith("#", StringComparison.Ordinal)) continue;
            // Use the (char[], StringSplitOptions) overload so this compiles
            // on net471 too — the (char, StringSplitOptions) overload is
            // net5+ only.
            var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 2)
            {
                throw new InvalidDataException(
                    $"Invalid merges.txt entry at {mergesPath}:{lineNumber}: '{rawLine}'. " +
                    $"Expected exactly two space-separated tokens.");
            }
            merges[(parts[0], parts[1])] = priority++;
        }

        _vocab = vocab;
        _inverseVocab = vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        _merges = merges;
    }

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
    /// Decodes a sequence of token IDs back to text.
    /// </summary>
    /// <param name="tokenIds">The token IDs to decode.</param>
    /// <returns>The decoded UTF-8 text.</returns>
    /// <remarks>
    /// <para>
    /// With a vocab loaded (<see cref="LoadVocab"/> or the
    /// <see cref="WhisperTokenizer(string, string)"/> ctor): each token ID is
    /// looked up in the inverse vocabulary to get its byte-unicode string,
    /// the strings are concatenated, each byte-unicode character is mapped
    /// back to its underlying byte (Radford 2019 GPT-2 byte mapping), and
    /// the resulting byte array is UTF-8 decoded. This is the exact inverse
    /// of <see cref="Encode"/> and matches HuggingFace's
    /// <c>WhisperTokenizer.decode</c>.
    /// </para>
    /// <para>
    /// Without a vocab loaded: tokens in <c>[0, 255]</c> are interpreted as
    /// raw UTF-8 byte values (the byte-level base layer that Encode emits in
    /// fallback mode); other token IDs are silently dropped because there's
    /// no merge table to expand them. This preserves the
    /// <c>Encode → Decode</c> round-trip for fallback-mode output.
    /// </para>
    /// </remarks>
    public string Decode(IEnumerable<long> tokenIds)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));

        if (_inverseVocab is null)
        {
            // Fallback: byte-level identity decode.
            var rawBytes = new List<byte>();
            foreach (long t in tokenIds)
            {
                int id = (int)t;
                if (id < 0 || IsSpecialToken(id)) continue;
                if (id <= 255) rawBytes.Add((byte)id);
            }
            return Encoding.UTF8.GetString(rawBytes.ToArray());
        }

        // Real BPE decode: concat token strings, then byte-unicode → bytes → UTF-8.
        var concatenated = new StringBuilder();
        foreach (long t in tokenIds)
        {
            int id = (int)t;
            if (IsSpecialToken(id)) continue; // drop control tokens
            if (_inverseVocab.TryGetValue(id, out var tokenStr))
                concatenated.Append(tokenStr);
            // Unknown IDs are silently dropped — matches HuggingFace behavior.
        }

        var bytes = new List<byte>(concatenated.Length);
        foreach (char c in concatenated.ToString())
        {
            if (_unicodeToByte.TryGetValue(c, out int b))
                bytes.Add((byte)b);
            // chars that aren't in the byte-unicode map indicate corruption; drop them.
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    /// <summary>
    /// Encodes text to GPT-2 / Whisper BPE token IDs.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>The encoded token IDs.</returns>
    /// <remarks>
    /// <para>
    /// With a vocab loaded: full byte-level BPE per Radford 2019 GPT-2 §2.2
    /// (also used by Whisper, Radford 2023 §2.2). Pipeline:
    /// </para>
    /// <list type="number">
    /// <item><description>UTF-8 encode text → bytes → byte-unicode chars.</description></item>
    /// <item><description>Pre-tokenize with GPT-2's regex pattern (contractions, word runs, punctuation).</description></item>
    /// <item><description>For each pre-token: split into characters and repeatedly merge the highest-priority adjacent pair from the merges table until none apply.</description></item>
    /// <item><description>Look up each final BPE piece in the vocab.</description></item>
    /// </list>
    /// <para>
    /// Without a vocab loaded: byte-level fallback — each UTF-8 byte becomes
    /// one token in <c>[0, 255]</c>. Both modes round-trip exactly through
    /// <see cref="Decode"/>.
    /// </para>
    /// </remarks>
    public List<long> Encode(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));

        if (_vocab is null || _merges is null)
        {
            // Fallback: byte-level identity. Each UTF-8 byte → token ID.
            var rawBytes = Encoding.UTF8.GetBytes(text);
            var tokens = new List<long>(rawBytes.Length);
            foreach (byte b in rawBytes) tokens.Add(b);
            return tokens;
        }

        // Real BPE encode.
        var output = new List<long>();
        foreach (Match m in _preTokenPattern.Matches(text))
        {
            var piece = m.Value;
            if (piece.Length == 0) continue;

            // Step 1: bytes → byte-unicode chars (so non-printable / multi-byte
            // characters are representable as ordinary char runs).
            var bytes = Encoding.UTF8.GetBytes(piece);
            var chars = new char[bytes.Length];
            for (int i = 0; i < bytes.Length; i++)
                chars[i] = _byteToUnicode[bytes[i]];

            // Step 2: build the BPE symbol list (start with one symbol per char).
            var symbols = new List<string>(chars.Length);
            foreach (char c in chars) symbols.Add(c.ToString());

            // Step 3: iteratively merge the highest-priority adjacent pair until
            // none of the remaining pairs is in the merge table.
            ApplyBpeMerges(symbols, _merges);

            // Step 4: map each final symbol to its vocab ID. Unknown symbols
            // are decomposed into their byte-unicode chars (each individual
            // byte-unicode char is guaranteed to be in the vocab since the
            // base 256 entries are always present).
            foreach (var sym in symbols)
            {
                if (_vocab.TryGetValue(sym, out int id))
                {
                    output.Add(id);
                }
                else
                {
                    foreach (char c in sym)
                    {
                        if (_vocab.TryGetValue(c.ToString(), out int charId))
                            output.Add(charId);
                    }
                }
            }
        }

        return output;
    }

    private static void ApplyBpeMerges(List<string> symbols, Dictionary<(string, string), int> merges)
    {
        while (symbols.Count >= 2)
        {
            // Find the lowest-rank merge among adjacent pairs.
            int bestPriority = int.MaxValue;
            int bestPosition = -1;
            for (int i = 0; i + 1 < symbols.Count; i++)
            {
                if (merges.TryGetValue((symbols[i], symbols[i + 1]), out int p) && p < bestPriority)
                {
                    bestPriority = p;
                    bestPosition = i;
                }
            }
            if (bestPosition < 0) break; // no more applicable merges

            symbols[bestPosition] = symbols[bestPosition] + symbols[bestPosition + 1];
            symbols.RemoveAt(bestPosition + 1);
        }
    }

    /// <summary>
    /// Builds the GPT-2 byte-to-unicode mapping (Radford 2019 §2.2). Bytes 33-126,
    /// 161-172, and 174-255 map to themselves as Unicode code points (printable
    /// ASCII / Latin-1); the remaining 68 bytes are assigned code points in the
    /// 256+ range so every byte has a unique printable representation. This
    /// representation is what vocab.json keys are written in.
    /// </summary>
    private static (Dictionary<int, char> ByteToChar, Dictionary<char, int> CharToByte) BuildByteUnicodeMaps()
    {
        var byteToChar = new Dictionary<int, char>();
        var charToByte = new Dictionary<char, int>();

        // "Native" bytes that map to themselves.
        var nativeBytes = new List<int>();
        for (int b = '!'; b <= '~'; b++) nativeBytes.Add(b);
        for (int b = '¡'; b <= '¬'; b++) nativeBytes.Add(b);
        for (int b = '®'; b <= 'ÿ'; b++) nativeBytes.Add(b);

        var nativeSet = new HashSet<int>(nativeBytes);
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (nativeSet.Contains(b))
            {
                byteToChar[b] = (char)b;
                charToByte[(char)b] = b;
            }
            else
            {
                // Map to a private-use-area-style point past 256.
                char c = (char)(256 + extra);
                byteToChar[b] = c;
                charToByte[c] = b;
                extra++;
            }
        }
        return (byteToChar, charToByte);
    }
}
