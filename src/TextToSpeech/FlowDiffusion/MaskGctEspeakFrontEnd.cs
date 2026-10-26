using System.Globalization;
using System.IO.Compression;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// MaskGCT's text front end for the languages it phonemizes with eSpeak NG (Amphion <c>models/tts/maskgct/g2p</c>):
/// English, French, German and Korean, each with the release's own normalization, phonemized through phonemizer's espeak
/// backend with MaskGCT's settings (phones split by "|", words by "|_|", punctuation <c>,.?!;:'…</c> kept), and the phone
/// vocabulary (<c>vocab.json</c>, 363 phones) that turns the result into the T2S model's ids.
/// </summary>
/// <remarks>eSpeak NG is the user's installed library (<see cref="EspeakNg"/>); MaskGCT's release used eSpeak NG 1.49.2.</remarks>
internal sealed class MaskGctEspeakFrontEnd
{
    /// <summary>MaskGCT's languages and the eSpeak voices its tokenizers use (<c>PhonemeBpeTokenizer.lang2backend</c>).</summary>
    public static readonly IReadOnlyDictionary<string, string> EspeakLanguages = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["zh"] = "cmn", ["ja"] = "ja", ["en"] = "en-us", ["fr"] = "fr-fr", ["ko"] = "ko", ["de"] = "de",
    };

    private const string Marks = ",.?!;:'…";
    private const string PhoneSeparator = "|";
    private const string WordSeparator = "|_|";

    private readonly Func<string, string, string>? _espeak;
    private readonly Dictionary<string, EspeakPhonemizer> _phonemizers = new(StringComparer.Ordinal);

    /// <param name="espeak">eSpeak's phonemes of (language, text); null for the installed library.</param>
    public MaskGctEspeakFrontEnd(Func<string, string, string>? espeak = null) => _espeak = espeak;

    // ---------------------------------------------------------------- the vocabulary

    private static readonly Lazy<Dictionary<string, int>> VocabularyInstance = new(() =>
    {
        using var stream = typeof(MaskGctEspeakFrontEnd).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.maskgct_vocab.json.gz")
            ?? throw new InvalidDataException("The embedded MaskGCT phone vocabulary is missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var vocab = (JObject)JObject.Parse(reader.ReadToEnd())["vocab"]!;
        return vocab.Properties().ToDictionary(p => p.Name, p => (int)p.Value, StringComparer.Ordinal);
    }, isThreadSafe: true);

    /// <summary>The phone vocabulary (<c>vocab.json</c>'s "vocab").</summary>
    public static IReadOnlyDictionary<string, int> Vocabulary => VocabularyInstance.Value;

    /// <summary><c>phoneme2token</c>: the ids of the phones (the text before any tab, split at "|"), unknown phones
    /// skipped.</summary>
    public static int[] Tokens(string phonemes)
    {
        int tab = phonemes.IndexOf('\t');
        if (tab >= 0) phonemes = phonemes.Substring(0, tab);
        var vocabulary = VocabularyInstance.Value;
        return phonemes.Split('|').Where(vocabulary.ContainsKey).Select(p => vocabulary[p]).ToArray();
    }

    // ---------------------------------------------------------------- the tokenizer

    private static readonly (string From, string To)[] ChinesePunctuation =
    {
        ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("；", ";"), ("：", ":"), ("、", ","),
        ("‘", "'"), ("’", "'"), ("⋯", "…"), ("···", "…"), ("・・・", "…"), ("...", "…"),
    };

    // Python's \s (str.isspace), and the marks the tokenizer keeps.
    private const string Space = @"\s\x1c-\x1f";
    private static readonly Regex AroundMark = new(@"[" + Space + @"]*([,\.\?!;:'…])[" + Space + @"]*", RegexOptions.CultureInvariant);
    private static readonly Regex Spaces = new(@"[" + Space + @"]+", RegexOptions.CultureInvariant);
    private static readonly Regex MarkPhone = new(@"([,\.\?!;:'…])", RegexOptions.CultureInvariant);
    private static readonly Regex Bars = new(@"\|+", RegexOptions.CultureInvariant);

    // Python's \w on str: letters and numbers of every script, and '_'.
    private static bool IsPythonWord(string s, int i)
    {
        var category = CharUnicodeInfo.GetUnicodeCategory(s, i);
        return s[i] == '_' || category <= UnicodeCategory.OtherLetter || category is UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber
            or UnicodeCategory.OtherNumber;
    }

    // re.sub(r"[^\w\s_,\.\?!;:\'…]", "", line), code point by code point.
    private static string KeepTokenizerCharacters(string line)
    {
        var sb = new StringBuilder(line.Length);
        for (int i = 0; i < line.Length; i += char.IsSurrogatePair(line, i) ? 2 : 1)
        {
            bool keep = IsPythonWord(line, i) || PythonText.IsSpace(line[i]) || Marks.IndexOf(line[i]) >= 0;
            if (keep) sb.Append(line, i, char.IsSurrogatePair(line, i) ? 2 : 1);
        }
        return sb.ToString();
    }

    /// <summary>MaskGCT's <c>TextTokenizer.__call__</c> for one text: each line stripped, its Chinese punctuation made ASCII,
    /// characters other than word characters, spaces and the kept marks dropped, spaces around marks removed and runs of
    /// spaces collapsed; then phonemized, each mark set off by "|" and repeated "|" collapsed, trailing "|" removed.</summary>
    public string Tokenize(string text, string espeakLanguage)
    {
        if (!_phonemizers.TryGetValue(espeakLanguage, out var phonemizer))
            _phonemizers[espeakLanguage] = phonemizer = new EspeakPhonemizer(espeakLanguage, Marks, _espeak);
        var lines = text.Trim('\n').Split('\n').Select(line =>
        {
            line = PythonText.Strip(line);
            foreach (var (from, to) in ChinesePunctuation) line = line.Replace(from, to);
            line = KeepTokenizerCharacters(line);
            line = AroundMark.Replace(line, "$1");
            return Spaces.Replace(line, " ");
        });
        string phonemized = string.Join("\n", lines.Select(l => phonemizer.Phonemize(l, PhoneSeparator, WordSeparator)));
        phonemized = Bars.Replace(MarkPhone.Replace(phonemized, "|$1|"), "|");
        return phonemized.TrimEnd('|');
    }

    // ---------------------------------------------------------------- English (english.py)

    private static readonly (Regex Pattern, string Replacement)[] EnglishAbbreviations = new[]
    {
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"), ("jr", "junior"), ("maj", "major"),
        ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"),
        ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"), ("etc", "et cetera"),
        ("btw", "by the way"),
    }.Select(x => (new Regex(@"\b" + x.Item1 + @"\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant), x.Item2)).ToArray();

    private static readonly Regex CommaNumber = new(@"([0-9][0-9,]+[0-9])", RegexOptions.CultureInvariant);
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex PercentNumber = new(@"([0-9\.,]*[0-9]+%)", RegexOptions.CultureInvariant);
    private static readonly Regex Pounds = new(@"£([0-9,]*[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Dollars = new(@"\$([0-9\.,]*[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Fraction = new(@"([0-9]+)/([0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Ordinal = new(@"[0-9]+(st|nd|rd|th)", RegexOptions.CultureInvariant);
    private static readonly Regex Number = new(@"[0-9]+", RegexOptions.CultureInvariant);

    private static string Int(string digits) => BigInteger.Parse(digits.Length == 0 ? "0" : digits, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);

    private static string ExpandDollars(Match m)
    {
        string match = m.Groups[1].Value;
        var parts = match.Split('.');
        if (parts.Length > 2) return " " + match + " dollars ";
        var dollars = BigInteger.Parse(parts[0].Length > 0 ? parts[0] : "0", CultureInfo.InvariantCulture);
        var cents = parts.Length > 1 && parts[1].Length > 0 ? BigInteger.Parse(parts[1], CultureInfo.InvariantCulture) : BigInteger.Zero;
        string Unit(BigInteger n, string one, string many) => n == BigInteger.One ? one : many;
        if (!dollars.IsZero && !cents.IsZero) return $" {dollars} {Unit(dollars, "dollar", "dollars")}, {cents} {Unit(cents, "cent", "cents")} ";
        if (!dollars.IsZero) return $" {dollars} {Unit(dollars, "dollar", "dollars")} ";
        if (!cents.IsZero) return $" {cents} {Unit(cents, "cent", "cents")} ";
        return " zero dollars ";
    }

    private static string FractionToWords(BigInteger numerator, BigInteger denominator)
    {
        if (numerator == 1 && denominator == 2) return " one half ";
        if (numerator == 1 && denominator == 4) return " one quarter ";
        string n = InflectNumberWords.NumberToWords(numerator.ToString(CultureInfo.InvariantCulture));
        if (denominator == 2) return " " + n + " halves ";
        if (denominator == 4) return " " + n + " quarters ";
        return " " + n + " " + InflectNumberWords.Ordinal(InflectNumberWords.NumberToWords(denominator.ToString(CultureInfo.InvariantCulture))) + " ";
    }

    private static string ExpandNumber(Match m)
    {
        var num = BigInteger.Parse(m.Value, CultureInfo.InvariantCulture);
        string W(BigInteger n) => InflectNumberWords.NumberToWords(n.ToString(CultureInfo.InvariantCulture));
        if (num > 1000 && num < 3000)
        {
            if (num == 2000) return " two thousand ";
            if (num > 2000 && num < 2010) return " two thousand " + W(num % 100) + " ";
            if (num % 100 == 0) return " " + W(num / 100) + " hundred ";
            return " " + InflectNumberWords.NumberToWords(num.ToString(CultureInfo.InvariantCulture), group: 2, andword: "", zero: "oh").Replace(", ", " ") + " ";
        }
        return " " + InflectNumberWords.NumberToWords(num.ToString(CultureInfo.InvariantCulture), andword: "") + " ";
    }

    /// <summary><c>normalize_numbers</c>: commas dropped from numbers; pounds, dollars and cents, fractions, decimals,
    /// percentages and ordinals in words; other numbers through inflect (years from 1001 to 2999 read in pairs).</summary>
    public static string NormalizeNumbers(string text)
    {
        text = CommaNumber.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = Pounds.Replace(text, "$1 pounds");
        text = Dollars.Replace(text, ExpandDollars);
        text = Fraction.Replace(text, m => FractionToWords(BigInteger.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture),
            BigInteger.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture)));
        text = DecimalNumber.Replace(text, m => m.Groups[1].Value.Replace(".", " point "));
        text = PercentNumber.Replace(text, m => m.Groups[1].Value.Replace("%", " percent "));
        text = Ordinal.Replace(text, m => " " + InflectNumberWords.NumberToWords(m.Value) + " ");
        return Number.Replace(text, ExpandNumber);
    }

    private static readonly (string Pattern, string Replacement)[] SpecialMap =
    {
        ("t|ɹ", "tɹ"), ("d|ɹ", "dɹ"), ("t|s", "ts"), ("d|z", "dz"), ("ɪ|ɹ", "ɪɹ"), ("ɐ", "ɚ"),
        ("ᵻ", "ɪ"), ("əl", "l"), ("x", "k"), ("ɬ", "l"), ("ʔ", "t"), ("n̩", "n"), ("oː|ɹ", "oːɹ"),
    };

    private static readonly (Regex Pattern, string Replacement)[] SpecialMapRegexes = SpecialMap
        .Select(x => (new Regex("(^|[_|])" + x.Pattern.Replace("|", @"\|") + "([_|]|$)", RegexOptions.CultureInvariant), "${1}" + x.Replacement + "${2}"))
        .ToArray();

    /// <summary><c>special_map</c>: phone merges and substitutions, each applied until it no longer matches.</summary>
    public static string ApplySpecialMap(string text)
    {
        foreach (var (pattern, replacement) in SpecialMapRegexes)
            while (pattern.IsMatch(text)) text = pattern.Replace(text, replacement);
        return text;
    }

    private const string EnglishFinalPhones = "p⁼ʰmftnlkxʃs`ɹaoəɛɪeɑʊŋiuɥwæjː";

    /// <summary><c>english_to_ipa</c>: abbreviations and numbers expanded, phonemized as en-us, "|_" appended after a final
    /// phone in the release's list, and the special map applied.</summary>
    public string English(string text)
    {
        foreach (var (pattern, replacement) in EnglishAbbreviations) text = pattern.Replace(text, replacement);
        string phonemes = Tokenize(NormalizeNumbers(text), EspeakLanguages["en"]);
        if (phonemes.Length == 0) throw new ArgumentException("The text has no phonemes.", nameof(text));
        if (EnglishFinalPhones.IndexOf(phonemes[phonemes.Length - 1]) >= 0) phonemes += "|_";
        return ApplySpecialMap(phonemes);
    }

    // ---------------------------------------------------------------- French and German (french.py, german.py)

    private static readonly (Regex Pattern, string Replacement)[] FrenchAbbreviations =
        new[]
        {
            ("M", "monsieur"), ("Mlle", "mademoiselle"), ("Mlles", "mesdemoiselles"), ("Mme", "Madame"), ("Mmes", "Mesdames"),
            ("N.B", "nota bene"), ("M", "monsieur"), ("p.c.q", "parce que"), ("Pr", "professeur"), ("qqch", "quelque chose"),
            ("rdv", "rendez-vous"), ("max", "maximum"), ("min", "minimum"), ("no", "numéro"), ("adr", "adresse"), ("dr", "docteur"),
            ("st", "saint"), ("co", "companie"), ("jr", "junior"), ("sgt", "sergent"), ("capt", "capitain"), ("col", "colonel"),
            ("av", "avenue"), ("av. J.-C", "avant Jésus-Christ"), ("apr. J.-C", "après Jésus-Christ"), ("art", "article"),
            ("boul", "boulevard"), ("c.-à-d", "c’est-à-dire"), ("etc", "et cetera"), ("ex", "exemple"), ("excl", "exclusivement"),
            ("boul", "boulevard"),
        }.Select(x => (new Regex(@"\b" + x.Item1 + @"\.", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant), x.Item2))
        .Concat(new[] { ("Mlle", "mademoiselle"), ("Mlles", "mesdemoiselles"), ("Mme", "Madame"), ("Mmes", "Mesdames") }
            .Select(x => (new Regex(@"\b" + x.Item1, RegexOptions.CultureInvariant), x.Item2)))
        .ToArray();

    private static readonly (string From, string To)[] RomancePunctuation =
    {
        ("：", ","), ("；", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", "."), ("·", ","),
        ("、", ","), ("...", "."), ("…", "."), ("$", "."), ("“", ""), ("”", ""), ("‘", ""), ("’", ""), ("（", ""),
        ("）", ""), ("(", ""), (")", ""), ("《", ""), ("》", ""), ("【", ""), ("】", ""), ("[", ""), ("]", ""), ("—", ""),
        ("～", "-"), ("~", "-"), ("「", ""), ("」", ""), ("¿", ""), ("¡", ""),
    };

    private static readonly Regex RomancePunctuationPattern =
        new(string.Join("|", RomancePunctuation.Select(p => Regex.Escape(p.From))), RegexOptions.CultureInvariant);

    private static readonly Regex AuxiliarySymbols = new("[<>()\\[\\]\"«»]+", RegexOptions.CultureInvariant);
    private static readonly Regex LeadingPunctuation = new(@"^[,.!?]+", RegexOptions.CultureInvariant);

    private static string NormalizeRomance(string text, bool french)
    {
        if (french)
            foreach (var (pattern, replacement) in FrenchAbbreviations) text = pattern.Replace(text, replacement.Replace("$", "$$"));
        var map = RomancePunctuation.ToDictionary(p => p.From, p => p.To, StringComparer.Ordinal);
        text = RomancePunctuationPattern.Replace(text, m => map[m.Value]);
        text = text.Replace(";", ",").Replace("-", " ").Replace(":", ",");
        if (french) text = text.Replace("&", " et ");
        text = AuxiliarySymbols.Replace(text, "");
        text = LeadingPunctuation.Replace(text, "");
        return PythonText.Strip(Spaces.Replace(text, " "));
    }

    /// <summary><c>french_to_ipa</c>: abbreviations expanded, punctuation normalized, phonemized as fr-fr.</summary>
    public string French(string text) => Tokenize(NormalizeRomance(text, french: true), EspeakLanguages["fr"]);

    /// <summary><c>german_to_ipa</c>: punctuation normalized, phonemized as de.</summary>
    public string German(string text) => Tokenize(NormalizeRomance(text, french: false), EspeakLanguages["de"]);

    // ---------------------------------------------------------------- Korean (korean.py)

    private static readonly Regex HanCharacters = new(
        "[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]",
        RegexOptions.CultureInvariant);

    private static readonly Regex Latin = new("([A-Za-z]+)", RegexOptions.CultureInvariant);

    private static readonly Dictionary<string, string> KoreanEnglish = new(StringComparer.Ordinal)
    {
        ["KOREA"] = "코리아", ["IDOL"] = "아이돌", ["IT"] = "아이티", ["IQ"] = "아이큐", ["UP"] = "업",
        ["DOWN"] = "다운", ["PC"] = "피씨", ["CCTV"] = "씨씨티비", ["SNS"] = "에스엔에스",
        ["AI"] = "에이아이", ["CEO"] = "씨이오", ["A"] = "에이", ["B"] = "비", ["C"] = "씨",
        ["D"] = "디", ["E"] = "이", ["F"] = "에프", ["G"] = "지", ["H"] = "에이치", ["I"] = "아이",
        ["J"] = "제이", ["K"] = "케이", ["L"] = "엘", ["M"] = "엠", ["N"] = "엔", ["O"] = "오", ["P"] = "피",
        ["Q"] = "큐", ["R"] = "알", ["S"] = "에스", ["T"] = "티", ["U"] = "유", ["V"] = "브이",
        ["W"] = "더블유", ["X"] = "엑스", ["Y"] = "와이", ["Z"] = "제트",
    };

    /// <summary><c>korean_to_ipa</c>: Han characters removed, listed English words spelled in Hangul, lowercased,
    /// phonemized as ko.</summary>
    public string Korean(string text)
    {
        text = HanCharacters.Replace(PythonText.Strip(text), "");
        text = Latin.Replace(text, m => KoreanEnglish.TryGetValue(m.Value, out var hangul) ? hangul : m.Value);
        return Tokenize(PythonText.Lower(text), EspeakLanguages["ko"]);
    }
    // ---------------------------------------------------------------- the pipeline's entry (g2p_generation.py)

    /// <summary>
    /// MaskGCT's <c>g2p_</c> for a language: "zh" and "en" through <c>chn_eng_g2p</c> (the text cut into runs of Chinese
    /// characters and of Latin letters, other characters joining the run before them, each run phonemized in its language
    /// and followed by "|", a final English run ending in "_" losing that word separator and its id), the others through
    /// their cleaner. Returns the phone string and the vocabulary ids. <paramref name="chinese"/> phonemizes a Chinese run
    /// given the whole sentence (MaskGCT's Mandarin front end).
    /// </summary>
    public (string Phonemes, int[] Tokens) G2P(string text, string language, Func<string, string, string>? chinese = null)
    {
        switch (language)
        {
            case "zh":
            case "en":
                var all = new StringBuilder();
                var tokens = new List<int>();
                var segments = Segments(text);
                for (int i = 0; i < segments.Count; i++)
                {
                    var (segment, kind) = segments[i];
                    string phonemes = kind switch
                    {
                        "en" => English(segment),
                        "zh" => (chinese ?? throw new InvalidOperationException("Chinese text needs MaskGCT's Mandarin front end."))(segment, text),
                        _ => throw new ArgumentException($"Unknown language: {kind}", nameof(text)),
                    };
                    all.Append(phonemes).Append('|');
                    tokens.AddRange(Tokens(phonemes));
                    if (kind == "en" && i == segments.Count - 1 && all.Length >= 2 && all[all.Length - 2] == '_')
                    {
                        all.Length -= 2;
                        tokens.RemoveAt(tokens.Count - 1);
                    }
                }
                return (all.ToString(), tokens.ToArray());
            case "fr":
                return WithTokens(French(text));
            case "de":
                return WithTokens(German(text));
            case "ko":
                return WithTokens(Korean(text));
            default:
                throw new ArgumentException($"MaskGCT's eSpeak front end has no language \"{language}\".", nameof(language));
        }
    }

    private static (string, int[]) WithTokens(string phonemes) => (phonemes, Tokens(phonemes));

    /// <summary><c>get_segment</c>: runs of Chinese characters (U+4E00–U+9FA5) and of ASCII letters, each other character
    /// joining the current run (and a leading "other" run taking the kind of the first letter after it).</summary>
    public static List<(string Text, string Kind)> Segments(string text)
    {
        static string Kind(char c) => c >= '一' && c <= '龥' ? "zh" : (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ? "en" : "other";
        var segments = new List<(string, string)>();
        if (text.Length == 0)
        {
            segments.Add(("", ""));
            return segments;
        }
        var current = new StringBuilder();
        string kind = "";
        for (int i = 0; i < text.Length; i++)
        {
            string type = Kind(text[i]);
            if (i == 0)
            {
                current.Append(text[i]);
                kind = type;
            }
            else if (kind == "other")
            {
                current.Append(text[i]);
                if (type != kind) kind = type;
            }
            else if (type == kind || type == "other")
            {
                current.Append(text[i]);
            }
            else
            {
                segments.Add((current.ToString(), kind));
                current.Clear().Append(text[i]);
                kind = type;
            }
        }
        segments.Add((current.ToString(), kind));
        return segments;
    }
}
