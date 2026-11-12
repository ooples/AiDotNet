using System.Globalization;
using System.Text;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>
/// The multilingual Chatterbox's text front end (resemble-ai/chatterbox <c>mtl_tts.punc_norm</c> and
/// <c>MTLTokenizer.encode</c>): lower case (Python's), NFKD, the language's own step, the <c>[xx]</c> language token, then
/// the grapheme BPE with spaces as <c>[SPACE]</c>.
/// </summary>
/// <remarks>
/// The language steps: Chinese is segmented by pkuseg and each Han character written as its Cangjie code
/// (<c>[cj_x]…[cj_.]</c>); Japanese kanji become their hiragana reading (pykakasi), katakana stays; Korean syllables are
/// decomposed into conjoining jamo; Hebrew gets its niqqud (Dicta); Russian gets its stress marks.
/// </remarks>
internal sealed class ChatterboxMultilingualText
{
    /// <summary>The release's 23 languages (<c>SUPPORTED_LANGUAGES</c>).</summary>
    public static readonly IReadOnlyDictionary<string, string> SupportedLanguages = new Dictionary<string, string>
    {
        ["ar"] = "Arabic", ["da"] = "Danish", ["de"] = "German", ["el"] = "Greek", ["en"] = "English", ["es"] = "Spanish",
        ["fi"] = "Finnish", ["fr"] = "French", ["he"] = "Hebrew", ["hi"] = "Hindi", ["it"] = "Italian", ["ja"] = "Japanese",
        ["ko"] = "Korean", ["ms"] = "Malay", ["nl"] = "Dutch", ["no"] = "Norwegian", ["pl"] = "Polish", ["pt"] = "Portuguese",
        ["ru"] = "Russian", ["sv"] = "Swedish", ["sw"] = "Swahili", ["tr"] = "Turkish", ["zh"] = "Chinese",
    };

    private readonly ChatterboxTokenizer _tokenizer;
    private readonly Dictionary<string, string> _wordToCangjie = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<string>> _cangjieToWords = new(StringComparer.Ordinal);
    private readonly Func<Pkuseg> _segmenter;
    private readonly Func<string, string> _hebrew;
    private readonly Func<string, string> _russian;

    /// <param name="tokenizer">The grapheme tokenizer (<c>grapheme_mtl_merged_expanded_v1.json</c>).</param>
    /// <param name="cangjiePath">The release's <c>Cangjie5_TC.json</c>.</param>
    /// <param name="segmenter">pkuseg, created on first Chinese text.</param>
    /// <param name="hebrew">Adds Hebrew niqqud.</param>
    /// <param name="russian">Adds Russian stress marks.</param>
    public ChatterboxMultilingualText(ChatterboxTokenizer tokenizer, string cangjiePath, Func<Pkuseg> segmenter,
        Func<string, string> hebrew, Func<string, string> russian)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _segmenter = segmenter ?? throw new ArgumentNullException(nameof(segmenter));
        _hebrew = hebrew ?? throw new ArgumentNullException(nameof(hebrew));
        _russian = russian ?? throw new ArgumentNullException(nameof(russian));
        // ChineseCangjieConverter._load_cangjie_mapping: "glyph\tcode..." entries; a glyph keeps its last code, a code
        // lists its glyphs in order.
        foreach (var entry in JArray.Parse(File.ReadAllText(cangjiePath, Encoding.UTF8)).Values<string>())
        {
            if (entry is null) continue;
            var fields = entry.Split('\t');
            if (fields.Length < 2) continue;
            _wordToCangjie[fields[0]] = fields[1];
            if (!_cangjieToWords.TryGetValue(fields[1], out var words)) _cangjieToWords[fields[1]] = words = new List<string>();
            words.Add(fields[0]);
        }
    }

    /// <summary><c>mtl_tts.punc_norm</c>: the English cleanup, with the CJK sentence enders also accepted.</summary>
    public static string NormalizePunctuation(string text) => Chatterbox<double>.NormalizeText(text, new[] { ".", "!", "?", "-", ",", "\u3001", "\uFF0C", "\u3002", "\uFF1F", "\uFF01" });

    /// <summary><c>MTLTokenizer.encode(text, language_id)</c>.</summary>
    public int[] Encode(string text, string? language)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (language is not null && !SupportedLanguages.ContainsKey(language.ToLowerInvariant()))
            throw new ArgumentException($"Unsupported language '{language}'. Supported languages: {string.Join(", ", SupportedLanguages.Keys)}.", nameof(language));
        string txt = PythonString.Lower(text).Normalize(NormalizationForm.FormKD);
        switch (language)
        {
            case "zh": txt = Cangjie(txt); break;
            case "ja": txt = Hiragana(txt); break;
            case "he": txt = _hebrew(txt); break;
            case "ko": txt = Korean(txt); break;
            case "ru": txt = _russian(txt); break;
        }
        if (language is not null) txt = "[" + language.ToLowerInvariant() + "]" + txt;
        return _tokenizer.Encode(txt);
    }

    // ChineseCangjieConverter.__call__: pkuseg's words joined by spaces, each "Lo" glyph with a code as its code letters
    // (and its rank among the code's glyphs when not the first) closed by [cj_.].
    private string Cangjie(string text)
    {
        string full = string.Join(" ", _segmenter().Cut(text));
        var output = new StringBuilder();
        foreach (var glyph in ChineseText.Codepoints(full))
        {
            if (CharUnicodeInfo.GetUnicodeCategory(glyph, 0) == UnicodeCategory.OtherLetter && _wordToCangjie.TryGetValue(glyph, out var code))
            {
                int index = _cangjieToWords[code].IndexOf(glyph);
                foreach (char c in code + (index > 0 ? index.ToString(CultureInfo.InvariantCulture) : "")) output.Append("[cj_").Append(c).Append(']');
                output.Append("[cj_.]");
            }
            else
            {
                output.Append(glyph);
            }
        }
        return output.ToString();
    }

    private static bool IsKanji(int c) => c >= 19968 && c <= 40959;
    private static bool IsKatakana(int c) => c >= 12449 && c <= 12538;

    // hiragana_normalize: a segment with kanji becomes its hiragana (spaced before a leading は or へ); an all-katakana
    // segment stays; then NFKD.
    private static string Hiragana(string text)
    {
        var output = new StringBuilder();
        foreach (var (orig, _, hira) in Pykakasi.Default.Convert(text))
        {
            var codes = ChineseText.Codepoints(orig).Select(g => char.ConvertToUtf32(g, 0)).ToList();
            if (codes.Any(IsKanji))
            {
                output.Append(hira.Length > 0 && (hira[0] == (char)0x306F || hira[0] == (char)0x3078) ? " " + hira : hira);
            }
            else
            {
                output.Append(orig);
            }
        }
        return output.ToString().Normalize(NormalizationForm.FormKD);
    }

    // korean_normalize: each code point from U+AC00 to U+D7AF split by the Hangul formula into initial, medial and
    // (when present) final jamo; then Python's strip().
    private static string Korean(string text)
    {
        var output = new StringBuilder(text.Length * 3);
        foreach (char c in text)
        {
            if (c < (char)0xAC00 || c > (char)0xD7AF)
            {
                output.Append(c);
                continue;
            }
            int b = c - 0xAC00;
            output.Append((char)(0x1100 + b / (21 * 28))).Append((char)(0x1161 + b % (21 * 28) / 28));
            if (b % 28 > 0) output.Append((char)(0x11A7 + b % 28));
        }
        return PythonString.Strip(output.ToString());
    }
}
