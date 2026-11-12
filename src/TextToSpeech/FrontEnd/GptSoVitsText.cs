using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>How GPT-SoVITS reads the text's language (<c>inference_webui.dict_language</c>).</summary>
public enum GptSoVitsTextLanguage
{
    /// <summary>All Chinese (<c>all_zh</c>): the segmenter's pieces read as Chinese, Latin pieces as English.</summary>
    Chinese,

    /// <summary>All English (<c>en</c>).</summary>
    English,

    /// <summary>All Japanese (<c>all_ja</c>).</summary>
    Japanese,

    /// <summary>All Cantonese (<c>all_yue</c>; v2 and later).</summary>
    Cantonese,

    /// <summary>All Korean (<c>all_ko</c>; v2 and later).</summary>
    Korean,

    /// <summary>Chinese mixed with English (<c>zh</c>): non-English pieces read as Chinese.</summary>
    ChineseEnglish,

    /// <summary>Japanese mixed with English (<c>ja</c>).</summary>
    JapaneseEnglish,

    /// <summary>Cantonese mixed with English (<c>yue</c>).</summary>
    CantoneseEnglish,

    /// <summary>Korean mixed with English (<c>ko</c>).</summary>
    KoreanEnglish,

    /// <summary>Each piece in the language the segmenter detects (<c>auto</c>).</summary>
    Auto,

    /// <summary>As <see cref="Auto"/>, Chinese pieces read as Cantonese (<c>auto_yue</c>).</summary>
    AutoCantonese,
}

/// <summary>How GPT-SoVITS cuts the text before synthesis (<c>how_to_cut</c>).</summary>
public enum GptSoVitsTextCut
{
    /// <summary>No cut (不切).</summary>
    None,

    /// <summary>Every four sentences (凑四句一切, <c>cut1</c>; the interface's default).</summary>
    FourSentences,

    /// <summary>Runs of over 50 characters (凑50字一切, <c>cut2</c>).</summary>
    FiftyCharacters,

    /// <summary>At Chinese full stops (按中文句号。切, <c>cut3</c>).</summary>
    ChinesePeriod,

    /// <summary>At English full stops not between digits (按英文句号.切, <c>cut4</c>).</summary>
    EnglishPeriod,

    /// <summary>At every punctuation mark (按标点符号切, <c>cut5</c>).</summary>
    Punctuation,
}

/// <summary>
/// GPT-SoVITS's text processing (<c>inference_webui.py</c> and <c>text/cleaner.py</c>): the cuts that split the target
/// text into sentences, the language routing of <c>get_phones_and_bert</c>, <c>clean_text</c> (each language's normalizer
/// and G2P, the special silence marks, unknown phones as UNK), the symbol ids, and the BERT features (Chinese only; zeros
/// otherwise).
/// </summary>
internal sealed class GptSoVitsText
{
    private readonly Func<GptSoVitsLangSegmenter> _segmenter;
    private readonly Func<GptSoVitsJapaneseFrontEnd> _japanese;
    private readonly Func<List<string>, List<List<string?>>>? _g2pw;
    private readonly Func<string, IReadOnlyList<int>, double[][]>? _bert;

    /// <param name="segmenter">The language segmenter (fastText's lid.176.bin), made on first use.</param>
    /// <param name="japanese">The Japanese front end (Open JTalk's dictionary), made on first use.</param>
    /// <param name="g2pw">G2PW's readings of sentences (Chinese after v1).</param>
    /// <param name="bert">The Chinese BERT's per-character features of normalized text (<c>hidden_states[-3]</c>
    /// without [CLS] and [SEP]), each 1024 wide; null for zeros.</param>
    /// <param name="v1">Version v1's symbol table and languages (zh, ja, en).</param>
    public GptSoVitsText(Func<GptSoVitsLangSegmenter> segmenter, Func<GptSoVitsJapaneseFrontEnd> japanese,
        Func<List<string>, List<List<string?>>>? g2pw, Func<string, IReadOnlyList<int>, double[][]>? bert, bool v1)
    {
        _segmenter = segmenter;
        _japanese = japanese;
        _g2pw = g2pw;
        _bert = bert;
        V1 = v1;
    }

    public bool V1 { get; }

    /// <summary>The BERT features' width.</summary>
    public const int BertDim = 1024;

    /// <summary>The symbol table (symbols.py for v1, symbols2.py after).</summary>
    public IReadOnlyList<string> Symbols => V1 ? GptSoVitsChineseData.Instance.SymbolsV1 : GptSoVitsChineseData.Instance.SymbolsV2;

    private Dictionary<string, int>? _ids;

    /// <summary><c>cleaned_text_to_sequence</c>.</summary>
    public int[] Ids(IEnumerable<string> phones)
    {
        _ids ??= Symbols.Select((s, i) => (s, i)).GroupBy(p => p.s, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Last().i, StringComparer.Ordinal);
        return phones.Select(p => _ids.TryGetValue(p, out int id) ? id : throw new KeyNotFoundException($"'{p}' is not a GPT-SoVITS symbol.")).ToArray();
    }

    // ------------------------------------------------------------------ cuts

    /// <summary><c>splits</c>: the sentence-ending marks.</summary>
    internal static readonly HashSet<string> Splits = new(StringComparer.Ordinal)
    {
        "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…",
    };

    private static readonly HashSet<string> CutPunctuation = new(StringComparer.Ordinal) { "!", "?", "…", ",", ".", "-", " " };

    private static List<string> Chars(string text) => ChineseText.Codepoints(text);

    private static string StripNewlines(string text) => text.Trim('\n');

    private static bool AllIn(string item, HashSet<string> set) => Chars(item).All(set.Contains);

    /// <summary><c>split</c>: the text after each sentence mark, a full stop appended when it ends without one.</summary>
    internal static List<string> Split(string text)
    {
        text = text.Replace("……", "。").Replace("——", "，");
        var chars = Chars(text);
        if (chars.Count == 0 || !Splits.Contains(chars[chars.Count - 1])) chars.Add("。");
        var pieces = new List<string>();
        int tail = 0;
        for (int head = 0; head < chars.Count; head++)
            if (Splits.Contains(chars[head]))
            {
                pieces.Add(string.Concat(chars.Skip(tail).Take(head + 1 - tail)));
                tail = head + 1;
            }
        return pieces;
    }

    /// <summary>The cut's sentences joined by newlines (<c>cut1</c> to <c>cut5</c>).</summary>
    internal static string Cut(string text, GptSoVitsTextCut cut)
    {
        string input = StripNewlines(text);
        List<string> options;
        switch (cut)
        {
            case GptSoVitsTextCut.None:
                return text;
            case GptSoVitsTextCut.FourSentences:
            {
                var pieces = Split(input);
                var starts = Enumerable.Range(0, (pieces.Count + 3) / 4).Select(i => i * 4).ToList();
                if (starts.Count > 1)
                {
                    options = new List<string>();
                    // split_idx[-1] = None: the last group runs to the end.
                    for (int i = 0; i < starts.Count - 1; i++)
                        options.Add(string.Concat(pieces.Skip(starts[i]).Take(i == starts.Count - 2 ? pieces.Count - starts[i] : starts[i + 1] - starts[i])));
                }
                else options = new List<string> { input };
                options = options.Where(o => !AllIn(o, CutPunctuation)).ToList();
                return string.Join("\n", options);
            }
            case GptSoVitsTextCut.FiftyCharacters:
            {
                var pieces = Split(input);
                if (pieces.Count < 2) return input;
                options = new List<string>();
                int sum = 0;
                string current = "";
                foreach (var p in pieces)
                {
                    sum += Chars(p).Count;
                    current += p;
                    if (sum > 50)
                    {
                        sum = 0;
                        options.Add(current);
                        current = "";
                    }
                }
                if (current.Length > 0) options.Add(current);
                if (options.Count > 1 && Chars(options[options.Count - 1]).Count < 50)
                {
                    options[options.Count - 2] += options[options.Count - 1];
                    options.RemoveAt(options.Count - 1);
                }
                options = options.Where(o => !AllIn(o, CutPunctuation)).ToList();
                return string.Join("\n", options);
            }
            case GptSoVitsTextCut.ChinesePeriod:
                options = input.Trim('。').Split('。').Where(o => !AllIn(o, CutPunctuation)).ToList();
                return string.Join("\n", options);
            case GptSoVitsTextCut.EnglishPeriod:
                options = Regex.Split(input.Trim('.'), @"(?<!\d)\.(?!\d)").Where(o => !AllIn(o, CutPunctuation)).ToList();
                return string.Join("\n", options);
            case GptSoVitsTextCut.Punctuation:
            {
                var marks = new HashSet<string>(StringComparer.Ordinal)
                {
                    ",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "：", "…",
                };
                var chars = Chars(input);
                var merged = new List<string>();
                var items = new List<string>();
                for (int i = 0; i < chars.Count; i++)
                {
                    items.Add(chars[i]);
                    if (!marks.Contains(chars[i])) continue;
                    bool decimalPoint = chars[i] == "." && i > 0 && i < chars.Count - 1 && PythonText.IsDigit(chars[i - 1]) && PythonText.IsDigit(chars[i + 1]);
                    if (decimalPoint) continue;
                    merged.Add(string.Concat(items));
                    items.Clear();
                }
                if (items.Count > 0) merged.Add(string.Concat(items));
                return string.Join("\n", merged.Where(o => !AllIn(o, marks)));
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(cut));
        }
    }

    /// <summary>
    /// The target text as <c>get_tts_wav</c> divides it: stripped of newlines at the ends, cut, blank lines collapsed,
    /// split at newlines, empty entries dropped (<c>process_text</c>), and runs shorter than five characters merged
    /// (<c>merge_short_text_in_array</c>).
    /// </summary>
    public static List<string> Sentences(string text, GptSoVitsTextCut cut)
    {
        text = Cut(StripNewlines(text), cut);
        while (text.Contains("\n\n")) text = text.Replace("\n\n", "\n");
        var texts = text.Split('\n').ToList();
        if (texts.All(t => t is " " or "\n" or "")) throw new ArgumentException("The text has nothing to read.", nameof(text));
        texts = texts.Where(t => t is not (" " or "")).ToList();
        if (texts.Count < 2) return texts;
        var result = new List<string>();
        string current = "";
        foreach (var t in texts)
        {
            current += t;
            if (Chars(current).Count >= 5)
            {
                result.Add(current);
                current = "";
            }
        }
        if (current.Length > 0)
        {
            if (result.Count == 0) result.Add(current);
            else result[result.Count - 1] += current;
        }
        return result;
    }

    /// <summary>Appends 。 (or . for English) when the sentence does not end with a sentence mark.</summary>
    public static string EndSentence(string text, GptSoVitsTextLanguage language)
    {
        var chars = Chars(text);
        if (chars.Count > 0 && Splits.Contains(chars[chars.Count - 1])) return text;
        return text + (language == GptSoVitsTextLanguage.English ? "." : "。");
    }

    // ------------------------------------------------------------------ get_phones_and_bert

    private static readonly Regex Spaces = new(" {2,}");

    /// <summary>
    /// <c>get_phones_and_bert(text, language, version)</c>: the phone ids, the BERT features <c>[phones][1024]</c> and the
    /// normalized text, retried once with a leading full stop when fewer than six phones come out.
    /// </summary>
    public (int[] Phones, double[][] Bert, string Normalized) PhonesAndBert(string text, GptSoVitsTextLanguage language, bool final = false)
    {
        text = Spaces.Replace(text, " ");
        var pieces = new List<(string Lang, string Text)>();
        string Code(GptSoVitsTextLanguage l) => l switch
        {
            GptSoVitsTextLanguage.ChineseEnglish => "zh",
            GptSoVitsTextLanguage.JapaneseEnglish => "ja",
            GptSoVitsTextLanguage.CantoneseEnglish => "yue",
            GptSoVitsTextLanguage.KoreanEnglish => "ko",
            _ => throw new ArgumentOutOfRangeException(nameof(language)),
        };
        switch (language)
        {
            case GptSoVitsTextLanguage.Chinese:
                pieces.AddRange(_segmenter().GetTexts(text, "zh"));
                break;
            case GptSoVitsTextLanguage.Cantonese:
                pieces.AddRange(_segmenter().GetTexts(text, "zh").Select(p => (p.Lang == "zh" ? "yue" : p.Lang, p.Text)));
                break;
            case GptSoVitsTextLanguage.Japanese:
                pieces.AddRange(_segmenter().GetTexts(text, "ja"));
                break;
            case GptSoVitsTextLanguage.Korean:
                pieces.AddRange(_segmenter().GetTexts(text, "ko"));
                break;
            case GptSoVitsTextLanguage.English:
                pieces.Add(("en", text));
                break;
            case GptSoVitsTextLanguage.Auto:
                pieces.AddRange(_segmenter().GetTexts(text));
                break;
            case GptSoVitsTextLanguage.AutoCantonese:
                pieces.AddRange(_segmenter().GetTexts(text).Select(p => (p.Lang == "zh" ? "yue" : p.Lang, p.Text)));
                break;
            default:
                // Chinese, Japanese and Korean Han characters cannot be told apart: non-English pieces take the caller's language.
                foreach (var (lang, piece) in _segmenter().GetTexts(text))
                {
                    if (pieces.Count > 0 && ((lang == "en") == (pieces[pieces.Count - 1].Lang == "en")))
                    {
                        pieces[pieces.Count - 1] = (pieces[pieces.Count - 1].Lang, pieces[pieces.Count - 1].Text + piece);
                        continue;
                    }
                    pieces.Add((lang == "en" ? "en" : Code(language), piece));
                }
                break;
        }
        var phones = new List<int>();
        var bert = new List<double[]>();
        var normalized = new System.Text.StringBuilder();
        foreach (var (lang, piece) in pieces)
        {
            var (p, wordToPhones, norm) = CleanText(piece, lang);
            var ids = Ids(p);
            phones.AddRange(ids);
            normalized.Append(norm);
            if (lang == "zh" && _bert is not null)
            {
                var rows = _bert(norm, wordToPhones!);
                if (rows.Length != ids.Length) throw new InvalidOperationException($"BERT gave {rows.Length} rows for {ids.Length} phones.");
                bert.AddRange(rows);
            }
            else bert.AddRange(Enumerable.Range(0, ids.Length).Select(_ => new double[BertDim]));
        }
        if (!final && phones.Count < 6) return PhonesAndBert("." + text, language, final: true);
        return (phones.ToArray(), bert.ToArray(), normalized.ToString());
    }

    // ------------------------------------------------------------------ clean_text

    private static readonly (string Mark, string Symbol)[] Special = { ("￥", "SP2"), ("^", "SP3") };

    /// <summary>
    /// <c>clean_text(text, language, version)</c>: the phones, the phones per character (Chinese and Cantonese) and the
    /// normalized text. Languages the version lacks read a single space as English.
    /// </summary>
    public (List<string> Phones, List<int>? WordToPhones, string Normalized) CleanText(string text, string language)
    {
        var languages = V1 ? new[] { "zh", "ja", "en" } : new[] { "zh", "ja", "en", "ko", "yue" };
        if (!languages.Contains(language))
        {
            language = "en";
            text = " ";
        }
        foreach (var (mark, symbol) in Special)
            if (language == "zh" && text.Contains(mark)) return CleanSpecial(text, mark, symbol);
        var symbols = new HashSet<string>(Symbols, StringComparer.Ordinal);
        List<string> phones;
        List<int>? wordToPhones = null;
        string norm;
        switch (language)
        {
            case "zh":
            {
                norm = GptSoVitsChineseFrontEnd.Default.TextNormalize(text);
                (phones, wordToPhones) = GptSoVitsChineseFrontEnd.Default.G2P(norm, V1, _g2pw);
                CheckAlignment(phones, wordToPhones, norm);
                break;
            }
            case "yue":
            {
                norm = GptSoVitsCantoneseFrontEnd.TextNormalize(text);
                (phones, wordToPhones) = GptSoVitsCantoneseFrontEnd.G2P(norm);
                CheckAlignment(phones, wordToPhones, norm);
                break;
            }
            case "en":
                norm = GptSoVitsEnglishFrontEnd.Default.TextNormalize(text);
                phones = GptSoVitsEnglishFrontEnd.Default.G2P(norm);
                if (phones.Count < 4) phones.Insert(0, ",");
                break;
            case "ja":
                norm = GptSoVitsJapaneseFrontEnd.TextNormalize(text);
                phones = _japanese().G2P(norm);
                break;
            default:
                norm = text;
                phones = GptSoVitsKoreanFrontEnd.G2P(norm);
                break;
        }
        phones = phones.Select(p => symbols.Contains(p) ? p : "UNK").ToList();
        return (phones, wordToPhones, norm);
    }

    private static void CheckAlignment(List<string> phones, List<int> wordToPhones, string norm)
    {
        if (phones.Count != wordToPhones.Sum()) throw new InvalidOperationException("The front end's phones do not add up to its phones per character.");
        if (Chars(norm).Count != wordToPhones.Count) throw new InvalidOperationException("The front end gave a phone count for each character but the text's length differs.");
    }

    /// <summary><c>clean_special</c>: a Chinese silence mark read as a comma, then its phone replaced by the mark's
    /// symbol.</summary>
    private (List<string> Phones, List<int>? WordToPhones, string Normalized) CleanSpecial(string text, string mark, string symbol)
    {
        text = text.Replace(mark, ",");
        string norm = GptSoVitsChineseFrontEnd.Default.TextNormalize(text);
        var (phones, wordToPhones) = GptSoVitsChineseFrontEnd.Default.G2P(norm, V1, _g2pw);
        var symbols = new HashSet<string>(Symbols, StringComparer.Ordinal);
        if (phones.Any(p => !symbols.Contains(p))) throw new InvalidOperationException("The Chinese front end gave a phone outside the symbol table.");
        return (phones.Select(p => p == "," ? symbol : p).ToList(), wordToPhones, norm);
    }
}
