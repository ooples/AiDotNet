using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// IndexTTS's text normalizer (the release's <c>indextts/utils/front.py</c> <c>TextNormalizer</c>, as it runs on Linux):
/// pronunciation annotations <c>&lt;字|读音&gt;</c> protected; Chinese (or digit-only, e-mail or pinyin-toned) text
/// normalized by WeTextProcessing's Chinese grammars with person names, hyphenated technical terms and toned pinyin
/// protected, English text by its English grammars with technical terms protected; English "'s" contractions expanded
/// first, an optional glossary applied, and punctuation folded to a small set.
/// </summary>
internal sealed class IndexTtsTextNormalizer
{
    private static readonly RegexOptions Options = RegexOptions.CultureInvariant;
    private static readonly Regex Han = new("[一-鿿]", Options);
    private static readonly Regex Latin = new("[a-zA-Z]", Options);
    private static readonly Regex Email = new("^[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-zA-Z]+$", Options);

    /// <summary>pinyin + tone digit 1–5, e.g. xuan4, jve2, shang5 (not beta1 or voice2).</summary>
    internal static readonly Regex PinyinTone = new(
        "(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])",
        Options | RegexOptions.IgnoreCase);

    private static readonly Regex Name = new("[一-鿿]+(?:[-·—][一-鿿]+){1,2}", Options | RegexOptions.IgnoreCase);
    private static readonly Regex TechTerm = new("[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+", Options);
    private static readonly Regex Contraction = new("(what|where|who|which|how|t?here|it|s?he|that|this)'s", Options | RegexOptions.IgnoreCase);
    private static readonly Regex Annotation = new("<([^|>\n]+)\\|([^>\n]+)>", Options);
    private static readonly Regex TechPlaceholder = new("\\s*<H>\\s*", Options);
    private static readonly Regex JqxU = new("([jqx])[uü](n|e|an)*(\\d)", Options | RegexOptions.IgnoreCase);

    // char_rep_map, in its insertion order (the alternation tries the keys in that order at each position).
    private static readonly (string From, string To)[] CharMap =
    {
        ("：", ","), ("；", ","), (";", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", " "), ("·", "-"), ("、", ","),
        ("...", "…"), (",,,", "…"), ("，，，", "…"), ("……", "…"), ("“", "'"), ("”", "'"), ("\"", "'"), ("‘", "'"), ("’", "'"),
        ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"), ("《", "'"), ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"),
        ("—", "-"), ("～", "-"), ("~", "-"), ("「", "'"), ("」", "'"), (":", ","),
    };

    private static readonly Regex EnglishFold = Alternation(CharMap);
    private static readonly Regex ChineseFold = Alternation(new[] { (From: "$", To: ".") }.Concat(CharMap).ToArray());
    private static readonly Dictionary<string, string> EnglishMap = CharMap.ToDictionary(p => p.From, p => p.To, StringComparer.Ordinal);
    private static readonly Dictionary<string, string> ChineseMap = new[] { (From: "$", To: ".") }.Concat(CharMap).ToDictionary(p => p.From, p => p.To, StringComparer.Ordinal);

    private readonly WeTextNormalizer _chinese, _english;

    public IndexTtsTextNormalizer(bool enableGlossary = true)
    {
        EnableGlossary = enableGlossary;
        _chinese = WeTextNormalizer.IndexTtsChinese;
        _english = WeTextNormalizer.English;
    }

    /// <summary>Whether <see cref="Glossary"/> applies (the release builds its normalizer with it on).</summary>
    public bool EnableGlossary { get; }

    /// <summary>Term readings (<c>term_glossary</c>): a term's reading per language (<c>"zh"</c>, <c>"en"</c>), or one
    /// reading for both under the key <c>""</c>. Matching ignores case; longer terms apply first.</summary>
    public Dictionary<string, Dictionary<string, string>> Glossary { get; } = new(StringComparer.Ordinal);

    private static Regex Alternation(IEnumerable<(string From, string To)> map) =>
        new(string.Join("|", map.Select(p => Regex.Escape(p.From))), Options);

    /// <summary><c>use_chinese</c>: Chinese characters, no Latin letters, an e-mail address, or toned pinyin.</summary>
    public static bool UseChinese(string text)
    {
        if (Han.IsMatch(text) || !Latin.IsMatch(text) || Email.IsMatch(text)) return true;
        return PinyinTone.IsMatch(text);
    }

    /// <summary><c>normalize</c>.</summary>
    public string Normalize(string text)
    {
        var annotations = new List<(string Key, string Value)>();
        text = Annotation.Replace(text, m =>
        {
            string key = $"PRONPLACEHOLDER{Alpha(annotations.Count)}PRONPLACEHOLDER";
            annotations.Add((key, m.Value));
            return key;
        });
        string result;
        if (UseChinese(text))
        {
            text = Contraction.Replace(text, "$1 is");
            if (EnableGlossary) text = ApplyGlossary(text, "zh");
            var (replaced, terms) = SaveTechTerms(PythonRstrip(text));
            (replaced, var pinyin) = SavePlaceholders(replaced, PinyinTone, "pinyin");
            (replaced, var names) = SavePlaceholders(replaced, Name, "n");
            try
            {
                result = _chinese.Normalize(replaced);
            }
            catch (InvalidOperationException)
            {
                result = "";
            }
            result = Restore(result, names, "n", n => n);
            result = Restore(result, pinyin, "pinyin", CorrectPinyin);
            if (terms.Count > 0) result = TechPlaceholder.Replace(result, "-");
            result = ChineseFold.Replace(result, m => ChineseMap[m.Value]);
        }
        else
        {
            text = Contraction.Replace(text, "$1 is");
            if (EnableGlossary) text = ApplyGlossary(text, "en");
            var (replaced, terms) = SaveTechTerms(text);
            try
            {
                result = _english.Normalize(replaced);
                if (terms.Count > 0) result = TechPlaceholder.Replace(result, "-");
            }
            catch (InvalidOperationException)
            {
                result = text;
            }
            result = EnglishFold.Replace(result, m => EnglishMap[m.Value]);
        }
        foreach (var (key, value) in annotations) result = result.Replace(key, value);
        return result;
    }

    private static string Alpha(int n)
    {
        string s = "";
        while (true)
        {
            s = (char)('a' + n % 26) + s;
            n = n / 26 - 1;
            if (n < 0) return s;
        }
    }

    private static string PythonRstrip(string s)
    {
        int end = s.Length;
        while (end > 0 && PythonText.IsSpace(s[end - 1])) end--;
        return s.Substring(0, end);
    }

    /// <summary>Distinct matches (first occurrence order) replaced by <c>&lt;{tag}_a&gt;</c>, <c>&lt;{tag}_b&gt;</c>, ….</summary>
    private static (string Text, List<string> Saved) SavePlaceholders(string text, Regex pattern, string tag)
    {
        var saved = pattern.Matches(text).Cast<Match>().Select(m => m.Value).Distinct().ToList();
        for (int i = 0; i < saved.Count; i++) text = text.Replace(saved[i], $"<{tag}_{(char)('a' + i)}>");
        return (text, saved);
    }

    private static string Restore(string text, List<string> saved, string tag, Func<string, string> map)
    {
        for (int i = 0; i < saved.Count; i++) text = text.Replace($"<{tag}_{(char)('a' + i)}>", map(saved[i]));
        return text;
    }

    /// <summary><c>save_tech_terms</c>: the hyphens of each technical term (longest first) replaced by <c>&lt;H&gt;</c>.</summary>
    private static (string Text, List<string> Terms) SaveTechTerms(string text)
    {
        var terms = TechTerm.Matches(text).Cast<Match>().Select(m => m.Value).Distinct().OrderByDescending(t => t.Length).ToList();
        foreach (var term in terms) text = text.Replace(term, term.Replace("-", "<H>"));
        return (text, terms);
    }

    /// <summary><c>correct_pinyin</c>: after j, q and x, u/ü become v and the syllable is upper-cased.</summary>
    internal static string CorrectPinyin(string pinyin)
    {
        if ("jqxJQX".IndexOf(pinyin[0]) < 0) return pinyin;
        return JqxU.Replace(pinyin, "${1}v${2}${3}").ToUpperInvariant();
    }

    private string ApplyGlossary(string text, string language)
    {
        foreach (var term in Glossary.Keys.OrderByDescending(k => k.Length))
        {
            var readings = Glossary[term];
            string replacement = readings.TryGetValue(language, out var r) ? r : readings.TryGetValue("", out var both) ? both : term;
            text = Regex.Replace(text, Regex.Escape(term), _ => replacement, Options | RegexOptions.IgnoreCase);
        }
        return text;
    }
}

/// <summary>
/// IndexTTS's text tokenizer (the release's <c>TextTokenizer</c>): normalization, then <c>tokenize_by_CJK_char</c> (CJK
/// characters spaced apart, everything upper-cased), then the BPE model; and the split of a token sequence into segments
/// at sentence punctuation, commas or hyphens and a maximum length, with short neighbours merged.
/// </summary>
internal sealed class IndexTtsTokenizer
{
    private static readonly Regex Cjk = new(
        "([ᄀ-ᇿ⺀-꓏ꡀ-힯豈-﫿︰-﹏･-ￜ]|[\ud840-\ud87f][\udc00-\udfff])",
        RegexOptions.CultureInvariant);

    private static readonly string[] SentenceMarks = { ".", "!", "?", "▁.", "▁?", "▁..." };

    public IndexTtsTokenizer(SentencePieceModel model, IndexTtsTextNormalizer? normalizer)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        Normalizer = normalizer;
    }

    public SentencePieceModel Model { get; }
    public IndexTtsTextNormalizer? Normalizer { get; }

    /// <summary><c>tokenize_by_CJK_char</c>.</summary>
    internal static string SplitCjk(string line)
    {
        var parts = Cjk.Split(PythonText.Strip(line));
        return string.Join(" ", parts.Select(PythonText.Strip).Where(w => w.Length > 0).Select(w => w.ToUpperInvariant()));
    }

    /// <summary><c>tokenize</c>: the BPE pieces of a text.</summary>
    public List<string> Tokenize(string text)
    {
        if (text.Length == 0) return new List<string>();
        if (PythonText.Strip(text).Length != 1)
        {
            if (Normalizer is not null) text = Normalizer.Normalize(text);
            text = SplitCjk(text);
        }
        return Model.EncodePieces(text).Select(p => p.Piece).ToList();
    }

    public int[] ToIds(IEnumerable<string> pieces) => pieces.Select(Model.PieceToId).ToArray();

    /// <summary><c>split_segments</c>.</summary>
    public static List<List<string>> SplitSegments(IReadOnlyList<string> tokens, int maxTokens = 120, int quickStreamingTokens = 0) =>
        SplitByToken(tokens, SentenceMarks, maxTokens, quickStreamingTokens);

    /// <summary><c>split_segments_by_token</c>.</summary>
    internal static List<List<string>> SplitByToken(IReadOnlyList<string> tokens, IReadOnlyCollection<string> splitTokens, int maxTokens,
        int quickStreamingTokens)
    {
        if (tokens.Count == 0) return new List<List<string>>();
        var segments = new List<List<string>>();
        var current = new List<string>();
        int length = 0;
        bool commaSplit = splitTokens.Contains(",") || splitTokens.Contains("▁,");
        for (int i = 0; i < tokens.Count; i++)
        {
            string token = tokens[i];
            current.Add(token);
            length++;
            List<List<string>> sub;
            if (!commaSplit && (current.Contains(",") || current.Contains("▁,")))
            {
                sub = SplitByToken(current, new[] { ",", "▁," }, maxTokens, quickStreamingTokens);
            }
            else if (!splitTokens.Contains("-") && current.Contains("-"))
            {
                sub = SplitByToken(current, new[] { "-" }, maxTokens, quickStreamingTokens);
            }
            else if (length <= maxTokens)
            {
                if (splitTokens.Contains(token) && length > 2)
                {
                    // The release appends a following quote to this segment; its "i += 1" does not skip it in a
                    // Python for loop, so the quote also starts the next segment.
                    if (i < tokens.Count - 1 && (tokens[i + 1] == "'" || tokens[i + 1] == "▁'")) current.Add(tokens[i + 1]);
                    segments.Add(current);
                    current = new List<string>();
                    length = 0;
                }
                continue;
            }
            else
            {
                sub = new List<List<string>>();
                for (int j = 0; j < current.Count; j += maxTokens)
                    sub.Add(current.GetRange(j, Math.Min(maxTokens, current.Count - j)));
            }
            segments.AddRange(sub);
            current = new List<string>();
            length = 0;
        }
        if (length > 0) segments.Add(current);

        var merged = new List<List<string>>();
        int total = 0;
        foreach (var segment in segments)
        {
            total += segment.Count;
            if (segment.Count == 0) continue;
            if (merged.Count == 0) merged.Add(segment);
            else if (merged[^1].Count + segment.Count <= maxTokens && total > quickStreamingTokens) merged[^1] = merged[^1].Concat(segment).ToList();
            else if (merged[^1].Count + segment.Count <= maxTokens / 2.0) merged[^1] = merged[^1].Concat(segment).ToList();
            else merged.Add(segment);
        }
        return merged;
    }
}
