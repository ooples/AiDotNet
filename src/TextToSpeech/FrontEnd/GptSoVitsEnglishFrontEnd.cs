using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// wordsegment 1.3.1 (Grant Jenks, after Peter Norvig's "Natural Language Corpus Data"; Apache-2.0): the most probable
/// division of a run of letters into words under the unigram and bigram counts, as <c>wordsegment.segment</c> computes it.
/// </summary>
internal sealed class WordSegment
{
    private static readonly Lazy<WordSegment> SharedInstance = new(() => new WordSegment(), isThreadSafe: true);
    private const double Total = 1024908267229.0;
    private const int Limit = 24;
    private readonly Dictionary<string, double> _unigrams = new(StringComparer.Ordinal);
    private readonly Dictionary<string, double> _bigrams = new(StringComparer.Ordinal);

    public static WordSegment Default => SharedInstance.Value;

    private WordSegment()
    {
        void Load(string resource, Dictionary<string, double> into)
        {
            foreach (var line in ChineseText.ReadLines(resource))
            {
                int tab = line.IndexOf('\t');
                into[line.Substring(0, tab)] = double.Parse(line.Substring(tab + 1), CultureInfo.InvariantCulture);
            }
        }
        Load("wordsegment_unigrams.tsv.gz", _unigrams);
        Load("wordsegment_bigrams.tsv.gz", _bigrams);
    }

    private double Score(string word, string? previous = null)
    {
        if (previous is null)
            return _unigrams.TryGetValue(word, out double count) ? count / Total : 10.0 / (Total * Math.Pow(10, word.Length));
        if (_bigrams.TryGetValue(previous + " " + word, out double bigram) && _unigrams.ContainsKey(previous))
            return bigram / Total / Score(previous);
        return Score(word);
    }

    // Python's ordering of (score, words) tuples: the score, then the words as lists of strings.
    private static int Compare((double Score, List<string> Words) a, (double Score, List<string> Words) b)
    {
        int c = a.Score.CompareTo(b.Score);
        if (c != 0) return c;
        for (int i = 0; i < Math.Min(a.Words.Count, b.Words.Count); i++)
        {
            int w = string.CompareOrdinal(a.Words[i], b.Words[i]);
            if (w != 0) return w;
        }
        return a.Words.Count.CompareTo(b.Words.Count);
    }

    /// <summary><c>wordsegment.segment(text)</c>.</summary>
    public List<string> Segment(string text)
    {
        var memo = new Dictionary<(string Suffix, string Prefix), (double, List<string>)>();
        (double Score, List<string> Words) Search(string rest, string previous)
        {
            if (rest.Length == 0) return (0.0, new List<string>());
            (double Score, List<string> Words)? best = null;
            for (int pos = 1; pos <= Math.Min(rest.Length, Limit); pos++)
            {
                string prefix = rest.Substring(0, pos), suffix = rest.Substring(pos);
                double prefixScore = Math.Log10(Score(prefix, previous));
                if (!memo.TryGetValue((suffix, prefix), out var found)) memo[(suffix, prefix)] = found = Search(suffix, prefix);
                var candidate = (prefixScore + found.Item1, new List<string> { prefix }.Concat(found.Item2).ToList());
                if (best is null || Compare(candidate, best.Value) > 0) best = candidate;
            }
            return best!.Value;
        }
        var clean = new StringBuilder();
        foreach (char c in text.ToLowerInvariant())
            if (c is >= 'a' and <= 'z' or >= '0' and <= '9') clean.Append(c);
        string cleaned = clean.ToString();
        var result = new List<string>();
        string carried = "";
        for (int offset = 0; offset < cleaned.Length; offset += 250)
        {
            string chunk = cleaned.Substring(offset, Math.Min(250, cleaned.Length - offset));
            var words = Search(carried + chunk, "<s>").Words;
            int keep = Math.Max(0, words.Count - 5);
            carried = string.Concat(words.Skip(keep));
            result.AddRange(words.Take(keep));
        }
        result.AddRange(Search(carried, "<s>").Words);
        return result;
    }
}

/// <summary>
/// GPT-SoVITS's English front end (RVC-Boss/GPT-SoVITS at b4570b8, MIT; <c>text/english.py</c> with
/// <c>text/en_normalization/expend.py</c>): the punctuation map, its number, time, unit, currency, fraction and arithmetic
/// expansion (inflect), accents stripped, everything but letters and <c>' . , ? ! -</c> dropped, capital letters split
/// off; then NLTK's TweetTokenizer and part-of-speech tagger and per word: g2p_en's homographs (with GPT-SoVITS's "read"
/// and "complex"), GPT-SoVITS's extended CMU dictionary, its name dictionary for title-case words, letters read out for
/// words of three letters or fewer, the possessive split off, wordsegment's compound split, else g2p_en's prediction.
/// </summary>
internal sealed class GptSoVitsEnglishFrontEnd
{
    private static readonly Lazy<GptSoVitsEnglishFrontEnd> SharedInstance = new(() => new GptSoVitsEnglishFrontEnd(), isThreadSafe: true);
    private readonly Dictionary<string, string[]> _cmu = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string[]> _names = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (string[] First, string[] Second, string Pos)> _homographs = new(StringComparer.Ordinal);

    public static GptSoVitsEnglishFrontEnd Default => SharedInstance.Value;

    private GptSoVitsEnglishFrontEnd()
    {
        void Load(string resource, Dictionary<string, string[]> into)
        {
            foreach (var line in ChineseText.ReadLines(resource))
            {
                int tab = line.IndexOf('\t');
                string phones = line.Substring(tab + 1);
                into[line.Substring(0, tab)] = phones.Length == 0 ? Array.Empty<string>() : phones.Split(' ');
            }
        }
        Load("gptsovits_engdict.tsv.gz", _cmu);
        Load("gptsovits_namedict.tsv.gz", _names);
        foreach (var line in ChineseText.ReadLines("g2pen_homographs.tsv.gz"))
        {
            if (line.StartsWith("#", StringComparison.Ordinal)) continue;
            var parts = line.Trim().Split('|');
            if (parts.Length != 4) continue;
            _homographs[parts[0].ToLowerInvariant()] = (parts[1].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries),
                parts[2].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries), parts[3]);
        }
        _homographs["read"] = (new[] { "R", "IY1", "D" }, new[] { "R", "EH1", "D" }, "VBP");
        _homographs["complex"] = (new[] { "K", "AH0", "M", "P", "L", "EH1", "K", "S" }, new[] { "K", "AA1", "M", "P", "L", "EH0", "K", "S" }, "JJ");
    }

    // ------------------------------------------------------------------ text_normalize

    // rep_map's keys are escaped before they are joined, so each matches only as the literal string it is written as.
    private static readonly (string From, string To)[] Replacements =
    {
        ("[;:：，；]", ","), ("[\"’]", "'"), ("。", "."), ("！", "!"), ("？", "?"),
    };

    private static readonly Regex ReplacementPattern = new(string.Join("|", Replacements.Select(r => Regex.Escape(r.From))));
    private static readonly Regex RepeatedPunctuation = new(@"([!?…,.\-\s])([!?…,.\-])+");

    /// <summary><c>english.text_normalize</c>.</summary>
    public string TextNormalize(string text)
    {
        text = ReplacementPattern.Replace(text, m => Replacements.First(r => r.From == m.Value).To);
        text = Expand(text);
        return RepeatedPunctuation.Replace(text, "$1");
    }

    private static readonly Regex OrdinalNumber = new(@"\b([0-9]+)\. ");
    private static readonly Regex CommaNumber = new(@"([0-9][0-9\,]+[0-9])");
    private static readonly Regex Time = new(@"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b");
    private static readonly Regex Measurement = new(@"\b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b");
    private static readonly Regex PoundsStart = new(@"£([0-9\.\,]*[0-9]+)");
    private static readonly Regex PoundsEnd = new(@"([0-9\.\,]*[0-9]+)£");
    private static readonly Regex DollarsStart = new(@"\$([0-9\.\,]*[0-9]+)");
    private static readonly Regex DollarsEnd = new(@"([(0-9\.\,]*[0-9]+)\$");
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.\s*[0-9]+)");
    private static readonly Regex Fraction = new(@"([0-9]+/[0-9]+)");
    private static readonly Regex OrdinalSuffix = new(@"[0-9]+(st|nd|rd|th)");
    private static readonly Regex Number = new(@"[0-9]+");
    private const string Superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ";
    private static readonly string Operand = @"((-?)((\d+)(\.\d+)?[" + Superscripts + @"]*)|(\.\d+[" + Superscripts + @"]*)|([A-Za-z][" + Superscripts + @"]*))";
    private static readonly Regex Asmd = new(Operand + @"\s+([+\-×÷=])\s+" + Operand);
    private static readonly Regex Integer = new(@"(?:^|\s+)(-)(\d+)");
    private static readonly Regex Disallowed = new(@"[^ A-Za-z'.,?!\-]");
    private static readonly Regex ThatIs = new(@"i\.e\.", RegexOptions.IgnoreCase);
    private static readonly Regex ForExample = new(@"e\.g\.", RegexOptions.IgnoreCase);
    private static readonly Regex Capital = new(@"(?<!^)(?<![\s])([A-Z])");

    private static readonly Dictionary<string, (string One, string Many)> Measures = new(StringComparer.Ordinal)
    {
        ["m"] = ("meter", "meters"), ["km"] = ("kilometer", "kilometers"), ["km/h"] = ("kilometer per hour", "kilometers per hour"),
        ["ft"] = ("feet", "feet"), ["L"] = ("liter", "liters"), ["tbsp"] = ("tablespoon", "tablespoons"), ["tsp"] = ("teaspoon", "teaspoons"),
        ["h"] = ("hour", "hours"), ["min"] = ("minute", "minutes"), ["s"] = ("second", "seconds"),
        ["°C"] = ("degree celsius", "degrees celsius"), ["°F"] = ("degree fahrenheit", "degrees fahrenheit"),
    };

    /// <summary><c>expend.normalize</c>.</summary>
    internal static string Expand(string text)
    {
        text = OrdinalNumber.Replace(text, m => DigitOrdinal(m.Groups[1].Value) + ", ");
        while (Asmd.IsMatch(text)) text = Asmd.Replace(text, ReplaceAsmd);
        text = Integer.Replace(text, m => "negative " + m.Groups[2].Value);
        text = CommaNumber.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = Time.Replace(text, ExpandTime);
        text = Measurement.Replace(text, ExpandMeasurement);
        text = PoundsStart.Replace(text, m => ExpandCurrency(m.Groups[1].Value, "pound", "pounds", "penny", "pence"));
        text = PoundsEnd.Replace(text, m => ExpandCurrency(m.Groups[1].Value, "pound", "pounds", "penny", "pence"));
        text = DollarsStart.Replace(text, m => ExpandCurrency(m.Groups[1].Value, "dollar", "dollars", "cent", "cents"));
        text = DollarsEnd.Replace(text, m => ExpandCurrency(m.Groups[1].Value, "dollar", "dollars", "cent", "cents"));
        text = DecimalNumber.Replace(text, ExpandDecimal);
        text = Fraction.Replace(text, ExpandFraction);
        text = OrdinalSuffix.Replace(text, m => InflectNumberWords.NumberToWords(m.Value));
        text = Number.Replace(text, m => ExpandNumber(m.Value));
        text = new string(text.Normalize(NormalizationForm.FormD).Where(c => CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark).ToArray());
        text = text.Replace("%", " percent");
        text = Disallowed.Replace(text, "");
        text = ThatIs.Replace(text, "that is");
        text = ForExample.Replace(text, "for example");
        return Capital.Replace(text, " $1");
    }

    // inflect's ordinal() of a number written in digits: the digits and their suffix.
    private static string DigitOrdinal(string digits)
    {
        int n = (int)(System.Numerics.BigInteger.Parse(digits, CultureInfo.InvariantCulture) % 100);
        string suffix = n is 11 or 12 or 13 ? "th" : (n % 10) switch { 1 => "st", 2 => "nd", 3 => "rd", _ => "th" };
        return digits + suffix;
    }

    private static string ReplaceAsmd(Match m)
    {
        string op = m.Groups[8].Value switch { "+" => " plus ", "-" => " minus ", "×" => " times ", "÷" => " divided by ", _ => " Equals " };
        return m.Groups[1].Value + op + m.Groups[9].Value;
    }

    private static string ExpandTime(Match m)
    {
        int hours = int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture), minutes = int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
        string period = hours < 12 ? "a.m." : "p.m.";
        if (hours > 12) hours -= 12;
        string hour = InflectNumberWords.NumberToWords(hours.ToString(CultureInfo.InvariantCulture));
        if (minutes == 0) return $"{hour} o'clock {period}";
        return $"{hour} {InflectNumberWords.NumberToWords(minutes.ToString(CultureInfo.InvariantCulture))} {period}";
    }

    private static string ExpandMeasurement(Match m)
    {
        string sign = m.Groups[3].Value;
        var number = System.Numerics.BigInteger.Parse(m.Groups[1].Value.Replace(sign, "").Replace(".", ""), CultureInfo.InvariantCulture);
        bool one = !m.Groups[2].Success && number == 1;
        return m.Groups[1].Value.Replace(sign, " " + (one ? Measures[sign].One : Measures[sign].Many));
    }

    // _expand_pounds and _expand_dollars.
    private static string ExpandCurrency(string match, string unit, string units, string subunit, string subunits)
    {
        var parts = match.Split('.');
        if (parts.Length > 2) return match + " " + units;
        var whole = parts[0].Length > 0 ? System.Numerics.BigInteger.Parse(parts[0], CultureInfo.InvariantCulture) : 0;
        var fraction = parts.Length > 1 && parts[1].Length > 0 ? System.Numerics.BigInteger.Parse(parts[1].PadRight(2, '0'), CultureInfo.InvariantCulture) : 0;
        string Unit(System.Numerics.BigInteger n, string single, string plural) => n == 1 ? single : plural;
        if (!whole.IsZero && !fraction.IsZero) return $"{whole} {Unit(whole, unit, units)} and {fraction} {Unit(fraction, subunit, subunits)}";
        if (!whole.IsZero) return $"{whole} {Unit(whole, unit, units)}";
        if (!fraction.IsZero) return $"{fraction} {Unit(fraction, subunit, subunits)}";
        return "zero " + units;
    }

    private static string ExpandDecimal(Match m)
    {
        var parts = m.Groups[1].Value.Split('.');
        return parts[0] + " point " + string.Join(" ", parts[1].Select(c => c.ToString()));
    }

    private static string ExpandFraction(Match m)
    {
        var parts = m.Value.Split('/');
        var numerator = System.Numerics.BigInteger.Parse(parts[0], CultureInfo.InvariantCulture);
        var denominator = System.Numerics.BigInteger.Parse(parts[1], CultureInfo.InvariantCulture);
        string numeratorPart = InflectNumberWords.NumberToWords(numerator.ToString(CultureInfo.InvariantCulture));
        string denominatorPart;
        if (denominator == 2) denominatorPart = numerator == 1 ? "half" : "halves";
        else if (denominator == 1) return numeratorPart;
        else
        {
            denominatorPart = InflectNumberWords.Ordinal(InflectNumberWords.NumberToWords(denominator.ToString(CultureInfo.InvariantCulture)));
            if (numerator > 1) denominatorPart += "s";
        }
        return $"{numeratorPart} {denominatorPart}";
    }

    private static string ExpandNumber(string digits)
    {
        var num = System.Numerics.BigInteger.Parse(digits, CultureInfo.InvariantCulture);
        string text = num.ToString(CultureInfo.InvariantCulture);
        if (num > 1000 && num < 3000)
        {
            if (num == 2000) return "two thousand";
            if (num > 2000 && num < 2010) return "two thousand " + InflectNumberWords.NumberToWords((num % 100).ToString(CultureInfo.InvariantCulture));
            if (num % 100 == 0) return InflectNumberWords.NumberToWords((num / 100).ToString(CultureInfo.InvariantCulture)) + " hundred";
            return InflectNumberWords.NumberToWords(text, group: 2, andword: "", zero: "oh").Replace(", ", " ");
        }
        return InflectNumberWords.NumberToWords(text, andword: "");
    }

    // ------------------------------------------------------------------ TweetTokenizer

    // nltk.tokenize.casual.TweetTokenizer() (nltk 3.10.4) on normalized text. Of its alternatives only those that can
    // match letters, spaces and ' . , ? ! - remain: a naked domain ("end.next"), a word with apostrophes or dashes, a
    // word, an ellipsis, any other character. The rest (URLs with a scheme or path, emoticons, HTML tags, arrows,
    // handles, hashtags, e-mail addresses, emoji, flags, phone numbers and numbers) each need a character
    // text_normalize removes; the tokenizer refuses such text rather than tokenize it differently.
    private static readonly Regex TweetWord = new(
        @"(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\b/?(?!@)" +
        @"|(?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_])" +
        @"|(?:[\w_]+)" +
        @"|(?:\.(?:[^\S\r\n]*\.){1,})" +
        @"|(?:\S)", RegexOptions.IgnoreCase);
    private static readonly Regex Hang = new(@"([^\p{L}\p{N}])\1{3,}");
    private static readonly Regex NormalizedAlphabet = new(@"^[ A-Za-z'.,?!\-]*$");

    internal static List<string> TweetTokenize(string text)
    {
        if (!NormalizedAlphabet.IsMatch(text))
            throw new ArgumentException("GPT-SoVITS tokenizes English only after text_normalize (letters, spaces and ' . , ? ! -).", nameof(text));
        string safe = Hang.Replace(text, "$1$1$1");
        return TweetWord.Matches(safe).Cast<Match>().Select(m => m.Value).ToList();
    }

    // ------------------------------------------------------------------ g2p

    private static readonly Regex HasLetter = new("[a-z]");
    private static readonly Regex Possessive = new("^([a-z]+)('s)$");

    /// <summary><c>english.g2p(text)</c> of normalized text: the ARPAbet phones with "UNK" for g2p_en's unknown, and
    /// apostrophes as "-".</summary>
    public List<string> G2P(string text)
    {
        var symbols = new HashSet<string>(GptSoVitsChineseData.Instance.SymbolsV2, StringComparer.Ordinal);
        var phones = new List<string>();
        foreach (var ph in Pronounce(text))
        {
            if (ph is " " or "<pad>" or "UW" or "</s>" or "<s>") continue;
            string p = ph == "<unk>" ? "UNK" : ph;
            if (symbols.Contains(p)) phones.Add(p);
            else if (p == "'") phones.Add("-");
        }
        return phones;
    }

    // en_G2p.__call__
    private List<string> Pronounce(string text)
    {
        var words = TweetTokenize(text);
        var tags = words.Count > 0 ? AveragedPerceptronTagger.English.Tag(words) : new List<string>();
        var prons = new List<string>();
        for (int i = 0; i < words.Count; i++)
        {
            string original = words[i], word = original.ToLowerInvariant(), pos = tags[i];
            IEnumerable<string> pron;
            if (!HasLetter.IsMatch(word)) pron = new[] { word };
            else if (word.Length == 1) pron = original == "A" ? new[] { "EY1" } : Cmu(word);
            else if (_homographs.TryGetValue(word, out var h))
            {
                if (pos.StartsWith(h.Pos, StringComparison.Ordinal)) pron = h.First;
                else if (pos.Length < h.Pos.Length && pos == h.Pos.Substring(0, pos.Length)) pron = h.First;
                else pron = h.Second;
            }
            else pron = QueryWord(original);
            prons.AddRange(pron);
            prons.Add(" ");
        }
        if (prons.Count > 0) prons.RemoveAt(prons.Count - 1);
        return prons;
    }

    private string[] Cmu(string word)
        => _cmu.TryGetValue(word, out var phones) ? phones : throw new KeyNotFoundException($"'{word}' is not in GPT-SoVITS's dictionary.");

    // str.istitle: a cased character starts each run of cased characters, the rest of the run is lower case.
    private static bool IsTitle(string word)
    {
        bool cased = false, previousCased = false;
        foreach (char c in word)
        {
            if (char.IsUpper(c) || CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.TitlecaseLetter)
            {
                if (previousCased) return false;
                previousCased = cased = true;
            }
            else if (char.IsLower(c))
            {
                if (!previousCased) return false;
                previousCased = cased = true;
            }
            else previousCased = false;
        }
        return cased;
    }

    // en_G2p.qryword
    private List<string> QueryWord(string original)
    {
        string word = original.ToLowerInvariant();
        if (word.Length > 1 && _cmu.TryGetValue(word, out var known)) return known.ToList();
        if (IsTitle(original) && _names.TryGetValue(word, out var name)) return name.ToList();
        if (word.Length <= 3)
        {
            var letters = new List<string>();
            foreach (char w in word)
            {
                if (w == 'a') letters.Add("EY1");
                else if (!char.IsLetter(w)) letters.Add(w.ToString());
                else letters.AddRange(Cmu(w.ToString()));
            }
            return letters;
        }
        if (Possessive.IsMatch(word))
        {
            var phones = QueryWord(word.Substring(0, word.Length - 2));
            string last = phones[phones.Count - 1];
            if (last is "P" or "T" or "K" or "F" or "TH" or "HH") phones.Add("S");
            else if (last is "S" or "Z" or "SH" or "ZH" or "CH" or "JH") phones.AddRange(new[] { "AH0", "Z" });
            else phones.Add("Z");
            return phones;
        }
        var parts = WordSegment.Default.Segment(word);
        if (parts.Count == 1) return G2pEn.Default.Predict(word);
        return parts.SelectMany(QueryWord).ToList();
    }
}
