using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// gruut 2.2.3's (MIT) settings and data for Spanish (es-es) and French (fr-fr) with eSpeak phonemes, as MeloTTS
/// loads them: the break and punctuation sets of <c>get_es_settings</c>/<c>get_fr_settings</c>, the eSpeak lexicon,
/// the grapheme-to-phoneme CRF, French's part-of-speech CRF, Babel's number symbols, month names and currency symbols
/// (under the en_US_POSIX default locale MeloTTS runs with on Linux), and num2words for verbalization.
/// </summary>
internal sealed class GruutLanguage
{
    private static readonly Lazy<GruutLanguage> SpanishInstance = new(() => new GruutLanguage(isFrench: false), isThreadSafe: true);
    private static readonly Lazy<GruutLanguage> FrenchInstance = new(() => new GruutLanguage(isFrench: true), isThreadSafe: true);

    public static GruutLanguage Spanish => SpanishInstance.Value;
    public static GruutLanguage French => FrenchInstance.Value;

    private readonly Lazy<Dictionary<string, List<(string Role, string Phonemes)>>> _lexicon;
    private readonly Lazy<CrfSuiteTagger> _g2p;
    private readonly Lazy<CrfSuiteTagger>? _pos;
    private readonly Dictionary<string, int> _monthWords = new(StringComparer.Ordinal);

    private GruutLanguage(bool isFrench)
    {
        IsFrench = isFrench;
        string code = isFrench ? "fr" : "es";
        BeginPunctuations = isFrench ? new[] { "\"", "“", "«", "[", "(", "<", "„" } : new[] { "\"", "“", "«", "[", "(", "<", "¡", "¿" };
        EndPunctuations = new[] { "\"", "”", "»", "]", ")", ">" };
        BeginPattern = new Regex("^(" + string.Join("|", BeginPunctuations.Select(Regex.Escape)) + ")", RegexOptions.CultureInvariant);
        EndPattern = new Regex("(" + string.Join("|", EndPunctuations.Select(Regex.Escape)) + ")$", RegexOptions.CultureInvariant);
        string ws = "[" + PythonText.WhitespaceClass + "]";
        MajorBreakPattern = new Regex(@"((?:\.|\?|!)+(?:" + ws + "+|$))", RegexOptions.CultureInvariant);
        MinorBreakPattern = new Regex(@"((?:,|;|:|\.\.\.)(?:" + ws + "+|$))", RegexOptions.CultureInvariant);
        WordBreakPattern = new Regex("(?:-|_)", RegexOptions.CultureInvariant);
        DateFormat = isFrench ? "OMY" : "DMY";
        Numbers = isFrench ? Num2WordsFrench.Default : Num2WordsSpanish.Default;

        _lexicon = new Lazy<Dictionary<string, List<(string, string)>>>(() =>
        {
            var lexicon = new Dictionary<string, List<(string, string)>>(StringComparer.Ordinal);
            foreach (var line in ChineseText.ReadLines($"gruut_{code}_lexicon.tsv.gz"))
            {
                int a = line.IndexOf('\t'), b = line.IndexOf('\t', a + 1);
                string word = line.Substring(0, a);
                if (!lexicon.TryGetValue(word, out var rows)) lexicon[word] = rows = new List<(string, string)>(1);
                rows.Add((line.Substring(a + 1, b - a - 1), line.Substring(b + 1)));
            }
            return lexicon;
        }, isThreadSafe: true);
        _g2p = new Lazy<CrfSuiteTagger>(() => CrfSuiteTagger.Load($"gruut_{code}_g2p.json.gz"), isThreadSafe: true);
        if (isFrench) _pos = new Lazy<CrfSuiteTagger>(() => CrfSuiteTagger.Load("gruut_fr_pos.json.gz"), isThreadSafe: true);

        var locale = (JObject)JObject.Parse(string.Join("\n", ChineseText.ReadLines("gruut_locales.json.gz")))[isFrench ? "fr_FR" : "es_ES"]!;
        DecimalSymbol = (string)locale["decimal"]!;
        GroupSymbol = (string)locale["group"]!;
        Months = ((JArray)locale["months"]!).Select(t => (string)t!).ToArray();
        Currencies = ((JArray)locale["currencies"]!).Select(t => ((string)t[0]!, (string)t[1]!)).ToArray();
        CurrencyCodes = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (symbol, currency) in Currencies) CurrencyCodes[symbol] = currency;

        // dateparser's month words for the language (its locale data), matched case- and accent-insensitively.
        string[][] months = isFrench
            ? new[] { new[] { "janv", "janvier", "jan" }, new[] { "févr", "février", "fév" }, new[] { "mars" }, new[] { "avr", "avril" },
                new[] { "mai" }, new[] { "juin", "jun" }, new[] { "juil", "juillet", "jul" }, new[] { "août", "aoû" },
                new[] { "sept", "septembre", "sep" }, new[] { "oct", "octobre" }, new[] { "nov", "novembre" }, new[] { "déc", "décembre" } }
            : new[] { new[] { "ene", "enero" }, new[] { "feb", "febrero" }, new[] { "mar", "marzo" }, new[] { "abr", "abril" },
                new[] { "may", "mayo" }, new[] { "jun", "junio" }, new[] { "jul", "julio" }, new[] { "ago", "agosto" },
                new[] { "sept", "septiembre", "setiembre", "sep", "set" }, new[] { "oct", "octubre" }, new[] { "nov", "noviembre" },
                new[] { "dic", "diciembre" } };
        for (int m = 0; m < 12; m++)
            foreach (var word in months[m]) _monthWords[FoldMonth(word)] = m + 1;
    }

    public bool IsFrench { get; }
    public string[] BeginPunctuations { get; }
    public string[] EndPunctuations { get; }
    public Regex BeginPattern { get; }
    public Regex EndPattern { get; }
    public Regex MajorBreakPattern { get; }
    public Regex MinorBreakPattern { get; }
    public Regex WordBreakPattern { get; }
    public string DateFormat { get; }
    public Num2WordsBase Numbers { get; }
    public string DecimalSymbol { get; }
    public string GroupSymbol { get; }
    public string[] Months { get; }
    public (string Symbol, string Currency)[] Currencies { get; }
    public Dictionary<string, string> CurrencyCodes { get; }
    public string DefaultCurrency => "EUR";

    internal Dictionary<string, List<(string Role, string Phonemes)>> Lexicon => _lexicon.Value;

    // ------------------------------------------------------------------ numbers

    private static readonly string[] SpaceChars = { " ", " ", " " };

    /// <summary>
    /// Babel's <c>parse_decimal(text, locale)</c> (non-strict): a space standing in for a space-like group symbol,
    /// the group symbol dropped, the decimal symbol made a point, then Python's <c>Decimal</c>.
    /// </summary>
    public bool TryParseDecimal(string text, out PythonDecimal value)
    {
        if (SpaceChars.Contains(GroupSymbol) && !text.Contains(GroupSymbol) && SpaceChars.Any(text.Contains))
            foreach (var space in SpaceChars) text = text.Replace(space, GroupSymbol);
        return PythonDecimal.TryParse(text.Replace(GroupSymbol, "").Replace(DecimalSymbol, "."), out value);
    }

    // ------------------------------------------------------------------ dates

    private static string FoldMonth(string word)
    {
        var decomposed = word.ToLowerInvariant().Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(decomposed.Length);
        foreach (char c in decomposed)
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark) sb.Append(c);
        return sb.ToString();
    }

    private static readonly Regex CompactDate = new("^([0-9]{1,4})(\\p{L}+)([0-9]{1,4})$", RegexOptions.CultureInvariant);

    /// <summary>
    /// The dates dateparser (strict, the language's locale) finds in one word: a day, a month word and a year, run
    /// together. The first number is the day when it has at most two digits and lies in 1-31, the second then the
    /// year (two or four digits; two-digit years 69-99 are 19xx, 00-68 20xx); otherwise the first is the year and the
    /// second the day. Impossible dates fail. Forms dateparser reads relative to the current time ("12h") are not
    /// dates here (see docs/text-to-speech/paper-fidelity-decisions.md).
    /// </summary>
    public bool TryParseDate(string text, out DateTime date)
    {
        date = default;
        var m = CompactDate.Match(text);
        if (!m.Success || !_monthWords.TryGetValue(FoldMonth(m.Groups[2].Value), out int month)) return false;
        string first = m.Groups[1].Value, second = m.Groups[3].Value;
        int firstValue = int.Parse(first, CultureInfo.InvariantCulture);
        string dayText, yearText;
        if (first.Length <= 2 && firstValue >= 1 && firstValue <= 31)
        {
            dayText = first;
            yearText = second;
        }
        else
        {
            yearText = first;
            dayText = second;
        }
        if (dayText.Length > 2 || (yearText.Length != 2 && yearText.Length != 4)) return false;
        int day = int.Parse(dayText, CultureInfo.InvariantCulture);
        int year = int.Parse(yearText, CultureInfo.InvariantCulture);
        if (yearText.Length == 2) year += year >= 69 ? 1900 : 2000;
        if (year < 1 || day < 1 || day > DateTime.DaysInMonth(year, month)) return false;
        date = new DateTime(year, month, day);
        return true;
    }

    // ------------------------------------------------------------------ CRF models

    private static string Base64(string s) => Convert.ToBase64String(Encoding.UTF8.GetBytes(s));

    /// <summary>gruut's <c>GraphemesToPhonemes</c> guess: the CRF over the lower-cased, NFC-composed word's graphemes.</summary>
    public List<string> GuessPhonemes(string word)
    {
        var graphemes = ChineseText.Codepoints(PythonText.Lower(word).Normalize(NormalizationForm.FormC));
        var items = new List<IReadOnlyList<(string, double)>>(graphemes.Count);
        for (int i = 0; i < graphemes.Count; i++)
        {
            var features = new List<(string, double)> { ("bias", 1.0), ("grapheme:" + Base64(graphemes[i]), 1.0) };
            if (i == 0) features.Add(("begin", 1.0));
            for (int j = 1; j <= 3; j++)
                if (i >= j) features.Add(($"grapheme-{j}:" + Base64(graphemes[i - j]), 1.0));
            for (int j = 1; j <= 3; j++)
                if (i < graphemes.Count - j) features.Add(($"grapheme+{j}:" + Base64(graphemes[i + j]), 1.0));
            if (i == graphemes.Count - 1) features.Add(("end", 1.0));
            items.Add(features);
        }
        var phonemes = new List<string>();
        foreach (var coded in _g2p.Value.Tag(items))
            foreach (var p in Encoding.UTF8.GetString(Convert.FromBase64String(coded)).Split('|'))
                if (p != "_") phonemes.Add(p);
        return phonemes;
    }

    private const string AsciiPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    private static void LocalFeatures(List<(string, double)> features, string word, string prefix)
    {
        var cp = ChineseText.Codepoints(word);
        features.Add(($"{prefix}bias", 1.0));
        features.Add(($"{prefix}word:" + Base64(word), 1.0));
        features.Add(($"{prefix}len(word)", cp.Count));
        features.Add(($"{prefix}word.ispunctuation", AsciiPunctuation.Contains(word) ? 1.0 : 0.0));
        features.Add(($"{prefix}word.isdigit()", PythonText.IsDigit(word) ? 1.0 : 0.0));
        for (int i = 2; i <= 3; i++) features.Add(($"{prefix}word[:{i}]:" + string.Concat(cp.Take(i)), 1.0));
        for (int i = 2; i <= 3; i++) features.Add(($"{prefix}word[-{i}:]:" + string.Concat(cp.Skip(Math.Max(0, cp.Count - i))), 1.0));
    }

    /// <summary>gruut's French <c>PartOfSpeechTagger</c>: the CRF over each word with two words of context either side.</summary>
    public string[] TagPartsOfSpeech(IReadOnlyList<string> words)
    {
        if (_pos is null) throw new InvalidOperationException("Only French has a part-of-speech model.");
        var items = new List<IReadOnlyList<(string, double)>>(words.Count);
        for (int i = 0; i < words.Count; i++)
        {
            var features = new List<(string, double)>();
            LocalFeatures(features, words[i], "");
            if (i == 0) features.Add(("BOS", 1.0));
            if (i == words.Count - 1) features.Add(("EOS", 1.0));
            for (int j = 1; j <= 2; j++)
                if (i >= j) LocalFeatures(features, words[i - j], $"-{j}:");
            for (int j = 1; j <= 2; j++)
                if (i < words.Count - j) LocalFeatures(features, words[i + j], $"+{j}:");
            items.Add(features);
        }
        return _pos.Value.Tag(items);
    }
}

/// <summary>
/// gruut's <c>SqlitePhonemizer</c> over the eSpeak lexicon, with its cache. One session serves one phonemizer call:
/// identical words within it share one phoneme list (so French liaison, which appends to that list, shows on every
/// occurrence, as in a fresh MeloTTS process), and nothing carries over to the next call.
/// </summary>
internal sealed class GruutLexiconSession
{
    private readonly GruutLanguage _language;
    private readonly Dictionary<string, List<(string Role, List<string> Phonemes)>> _cache = new(StringComparer.Ordinal);

    public GruutLexiconSession(GruutLanguage language) => _language = language;

    // remove_non_word_chars: drop what Python's \W matches.
    private static string RemoveNonWordChars(string s)
    {
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            bool pair = char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]);
            var category = CharUnicodeInfo.GetUnicodeCategory(s, i);
            bool word = s[i] == '_' || category is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter
                or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter
                or UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber;
            if (word) sb.Append(pair ? s.Substring(i, 2) : s[i].ToString());
            if (pair) i++;
        }
        return sb.ToString();
    }

    /// <summary>The phonemes of a word (exact role, else the default role, else the first), or null.</summary>
    public List<string>? Lookup(string word, string? role, bool doTransforms = true)
    {
        word = PythonText.Lower(word);
        if (_cache.TryGetValue(word, out var cached))
        {
            if (role is not null)
                foreach (var (r, p) in cached)
                    if (r == role) return p;
            foreach (var (r, p) in cached)
                if (r.Length == 0) return p;
            return cached.Count > 0 ? cached[0].Phonemes : null;
        }
        var candidates = new List<string> { word };
        if (doTransforms)
        {
            candidates.Add(RemoveNonWordChars(word));
            candidates.Add(RemoveNonWordChars(PythonText.Lower(word)));
        }
        List<(string Role, List<string> Phonemes)>? roleToWord = null;
        foreach (var lookupWord in candidates)
        {
            if (lookupWord.Length == 0) continue;
            if (_language.Lexicon.TryGetValue(lookupWord, out var rows))
            {
                foreach (var (dbRole, dbPhonemes) in rows)
                {
                    if (roleToWord is null)
                    {
                        roleToWord = new List<(string, List<string>)>();
                        _cache[word] = roleToWord;
                    }
                    if (!roleToWord.Any(e => e.Role == dbRole))
                        roleToWord.Add((dbRole, dbPhonemes.Split(new[] { ' ', '\t', '\n', '\r', '\f', '\v' }, StringSplitOptions.RemoveEmptyEntries).ToList()));
                }
            }
            if (roleToWord is not null)
            {
                _cache[lookupWord] = _cache[word];
                return Lookup(word, role);
            }
        }
        return null;
    }
}
