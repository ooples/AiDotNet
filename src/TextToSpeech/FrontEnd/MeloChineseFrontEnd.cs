using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// pypinyin 0.50.0's <c>lazy_pinyin</c> in the INITIALS and FINALS_TONE3 styles (neutral tone as 5), the two MeloTTS
/// reads: Han runs cut by forward maximum matching on the phrase dictionary, each piece read as a phrase or character by
/// character, other runs kept as written (as both their initial and final).
/// </summary>
internal sealed class Pypinyin050
{
    private static readonly Lazy<Pypinyin050> SharedInstance = new(() => new Pypinyin050(), isThreadSafe: true);
    private readonly Dictionary<string, (string Initial, string Final)> _characters = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (string Initial, string Final)[]> _phrases = new(StringComparer.Ordinal);
    private readonly HashSet<string> _phrasePrefixes = new(StringComparer.Ordinal);

    public static Pypinyin050 Default => SharedInstance.Value;

    private static (string, string) Pair(string reading)
    {
        int comma = reading.IndexOf(',');
        return (reading.Substring(0, comma), reading.Substring(comma + 1));
    }

    private Pypinyin050()
    {
        foreach (var line in JiebaSegmenter.ReadLines("pypinyin050_chars.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            _characters[line.Substring(0, tab)] = Pair(line.Substring(tab + 1));
        }
        foreach (var line in JiebaSegmenter.ReadLines("pypinyin050_phrases.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            string phrase = line.Substring(0, tab);
            _phrases[phrase] = line.Substring(tab + 1).Split(' ').Select(Pair).ToArray();
            var prefix = new StringBuilder();
            foreach (var c in JiebaSegmenter.Codepoints(phrase))
            {
                prefix.Append(c);
                _phrasePrefixes.Add(prefix.ToString());
            }
        }
    }

    /// <summary>Each item's (INITIALS, FINALS_TONE3) reading: a syllable, or a run of non-Han text as written.</summary>
    public List<(string Initial, string Final)> LazyPinyin(string word)
    {
        var result = new List<(string, string)>();
        var runs = new List<(bool Han, List<string> Text)>();
        foreach (var c in JiebaSegmenter.Codepoints(word))
        {
            bool han = MandarinG2P.IsHan(c);
            if (runs.Count == 0 || runs[runs.Count - 1].Han != han) runs.Add((han, new List<string>()));
            runs[runs.Count - 1].Text.Add(c);
        }
        foreach (var (han, text) in runs)
        {
            if (!han)
            {
                string kept = string.Concat(text);
                result.Add((kept, kept));
                continue;
            }
            foreach (var piece in MaximumMatching(text))
            {
                if (piece.Count > 1 && _phrases.TryGetValue(string.Concat(piece), out var readings)) result.AddRange(readings);
                else result.AddRange(piece.Select(c => _characters.TryGetValue(c, out var r) ? r : (c, c)));
            }
        }
        return result;
    }

    // pypinyin.seg.mmseg.Seg(no_non_phrases=True).cut
    private IEnumerable<List<string>> MaximumMatching(List<string> text)
    {
        int start = 0;
        while (start < text.Count)
        {
            int lastValid = 0;
            bool broke = false;
            for (int index = start; index < text.Count; index++)
            {
                string candidate = string.Concat(text.GetRange(start, index - start + 1));
                if (_phrasePrefixes.Contains(candidate))
                {
                    if (_phrases.ContainsKey(candidate)) lastValid = index - start + 1;
                }
                else
                {
                    if (lastValid > 0)
                    {
                        yield return text.GetRange(start, lastValid);
                        start += lastValid;
                    }
                    else
                    {
                        yield return text.GetRange(start, 1);
                        start += 1;
                    }
                    broke = true;
                    break;
                }
            }
            if (broke) continue;
            if (lastValid > 0)
            {
                yield return text.GetRange(start, lastValid);
                start += lastValid;
            }
            else
            {
                string remain = string.Concat(text.GetRange(start, text.Count - start));
                if (_phrases.ContainsKey(remain)) yield return text.GetRange(start, text.Count - start);
                else
                    foreach (var c in text.GetRange(start, text.Count - start)) yield return new List<string> { c };
                yield break;
            }
        }
    }
}

/// <summary>
/// MeloTTS's Chinese front end (myshell-ai/MeloTTS <c>melo/text/chinese.py</c> with <c>tone_sandhi.py</c>): numerals
/// read as Chinese numerals (cn2an), punctuation mapped and everything outside Chinese and MeloTTS's punctuation dropped,
/// then per sentence: jieba's part-of-speech segmentation, the tone sandhi's word merges, pypinyin's initials and
/// finals, the sandhi's tone changes (不, 一, neutral tones, third-tone sandhi), and the opencpop phone table.
/// </summary>
internal sealed class MeloChineseFrontEnd
{
    private static readonly Lazy<MeloChineseFrontEnd> SharedInstance = new(() => new MeloChineseFrontEnd(), isThreadSafe: true);
    private readonly Dictionary<string, string> _opencpop = new(StringComparer.Ordinal);
    private readonly HashSet<string> _mustNeutral = new(StringComparer.Ordinal);
    private readonly HashSet<string> _mustNotNeutral = new(StringComparer.Ordinal);
    private readonly HashSet<string> _numeric = new(StringComparer.Ordinal);

    public static MeloChineseFrontEnd Default => SharedInstance.Value;

    private const string Punctuation = "!?…,.'-¿¡";

    private MeloChineseFrontEnd()
    {
        var data = JObject.Parse(string.Join("\n", JiebaSegmenter.ReadLines("melo_chinese.json.gz")));
        foreach (var entry in (JObject)data["opencpop"]!) _opencpop[entry.Key] = (string)entry.Value!;
        foreach (var w in (JArray)data["must_neural_tone_words"]!) _mustNeutral.Add((string)w!);
        foreach (var w in (JArray)data["must_not_neural_tone_words"]!) _mustNotNeutral.Add((string)w!);
        foreach (var c in JiebaSegmenter.Codepoints((string)data["numeric"]!)) _numeric.Add(c);
    }

    private bool IsNumeric(string c) => _numeric.Contains(c);

    // ------------------------------------------------------------------ normalization

    private static readonly Regex Numbers = new(@"\d+(?:\.?\d+)?", RegexOptions.CultureInvariant);
    private static readonly (string From, string To)[] Replacements =
    {
        ("：", ","), ("；", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", "."), ("·", ","), ("、", ","), ("...", "…"),
        ("$", "."), ("“", "'"), ("”", "'"), ("‘", "'"), ("’", "'"), ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"), ("《", "'"), ("》", "'"),
        ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"), ("—", "-"), ("～", "-"), ("~", "-"), ("「", "'"), ("」", "'"),
    };
    private static readonly Regex ReplacementPattern = new(string.Join("|", Replacements.Select(r => Regex.Escape(r.From))), RegexOptions.CultureInvariant);
    // As the reference builds it: inside the class, "'-¿" is a range (U+0027 to U+00BF), so ASCII letters and digits stay.
    private static readonly Regex Outside = new("[^一-龥" + Punctuation + "]+", RegexOptions.CultureInvariant);

    /// <summary><c>chinese.text_normalize</c>.</summary>
    public string Normalize(string text) => Outside.Replace(NumeralsAndPunctuation(text), "");

    /// <summary>
    /// The steps <c>chinese.text_normalize</c> shares with <c>chinese_mix.text_normalize</c>: each numeral read by cn2an
    /// (the first occurrence of each match, in turn), then 嗯/呣 and the punctuation map.
    /// </summary>
    internal static string NumeralsAndPunctuation(string text)
    {
        foreach (Match m in Numbers.Matches(text))
        {
            int at = text.IndexOf(m.Value, StringComparison.Ordinal);
            if (at >= 0) text = text.Substring(0, at) + MandarinG2P.AnToCn(m.Value) + text.Substring(at + m.Value.Length);
        }
        text = text.Replace("嗯", "恩").Replace("呣", "母");
        return ReplacementPattern.Replace(text, m => Replacements.First(r => r.From == m.Value).To);
    }

    // ------------------------------------------------------------------ G2P

    private static readonly Regex SentenceSplit = new("(?<=[" + Punctuation + "])\\s*", RegexOptions.CultureInvariant);
    private static readonly Regex Latin = new("[a-zA-Z]+", RegexOptions.CultureInvariant);

    /// <summary>
    /// <c>chinese.g2p(text)</c> of normalized text: the phones, their tones (0 for punctuation, 1–5), and the phones per
    /// character, with the silence symbol at each end.
    /// </summary>
    public (List<string> Phones, List<int> Tones, List<int> PhonesPerCharacter) G2P(string text)
    {
        var sentences = SentenceSplit.Split(text).Where(s => s.Trim().Length > 0).ToList();
        var phones = new List<string>();
        var tones = new List<int>();
        var perCharacter = new List<int>();
        foreach (var raw in sentences) AppendSegment(raw, phones, tones, perCharacter);
        // The reference asserts both ("Sometimes it will crash"): a character pypinyin or jieba drops breaks the alignment.
        if (perCharacter.Sum() != phones.Count || perCharacter.Count != JiebaSegmenter.Codepoints(text).Count)
            throw new ArgumentException($"The phones of '{text}' do not align one reading per character.", nameof(text));
        phones.Insert(0, "_");
        phones.Add("_");
        tones.Insert(0, 0);
        tones.Add(0);
        perCharacter.Insert(0, 1);
        perCharacter.Add(1);
        return (phones, tones, perCharacter);
    }

    /// <summary><c>chinese._g2p</c> of one segment, appended (no silences; tones before any language offset).</summary>
    internal void AppendSegment(string raw, List<string> phones, List<int> tones, List<int> perCharacter)
    {
        var initials = new List<string>();
        var finals = new List<string>();
        foreach (var (word, pos) in PreMerge(JiebaSegmenter.Default.PosCut(Latin.Replace(raw, ""))))
        {
            if (pos == "eng") continue;
            var readings = Pypinyin050.Default.LazyPinyin(word);
            var subFinals = ModifiedTone(word, pos, readings.Select(r => r.Final).ToList());
            initials.AddRange(readings.Select(r => r.Initial));
            finals.AddRange(subFinals);
        }
        for (int i = 0; i < initials.Count; i++)
        {
            string c = initials[i], v = finals[i];
            if (c == v)
            {
                if (!Punctuation.Contains(c) || c.Length != 1)
                    throw new ArgumentException($"'{c}' has no Chinese reading and is not MeloTTS punctuation.", nameof(raw));
                phones.Add(c);
                tones.Add(0);
                perCharacter.Add(1);
                continue;
            }
            string vWithoutTone = v.Substring(0, v.Length - 1);
            char tone = v[v.Length - 1];
            string pinyin = c + vWithoutTone;
            if ("12345".IndexOf(tone) < 0) throw new ArgumentException($"The reading {c}{v} has no tone.", nameof(raw));
            if (c.Length > 0)
            {
                string? replaced = vWithoutTone switch { "uei" => "ui", "iou" => "iu", "uen" => "un", _ => null };
                if (replaced is not null) pinyin = c + replaced;
            }
            else
            {
                string? whole = pinyin switch { "ing" => "ying", "i" => "yi", "in" => "yin", "u" => "wu", _ => null };
                if (whole is not null) pinyin = whole;
                else
                {
                    string? first = pinyin[0] switch { 'v' => "yu", 'e' => "e", 'i' => "y", 'u' => "w", _ => null };
                    if (first is not null) pinyin = first + pinyin.Substring(1);
                }
            }
            if (!_opencpop.TryGetValue(pinyin, out var symbols))
                throw new ArgumentException($"The syllable {pinyin} is not in MeloTTS's phone table.", nameof(raw));
            var syllable = symbols.Split(' ');
            phones.AddRange(syllable);
            tones.AddRange(Enumerable.Repeat(tone - '0', syllable.Length));
            perCharacter.Add(syllable.Length);
        }
    }

    /// <summary>One word's pypinyin initials and its finals after the tone sandhi (<c>modified_tone</c>).</summary>
    public (List<string> Initials, List<string> Finals) Readings(string word, string pos)
    {
        var readings = Pypinyin050.Default.LazyPinyin(word);
        return (readings.Select(r => r.Initial).ToList(), ModifiedTone(word, pos, readings.Select(r => r.Final).ToList()));
    }

    // ------------------------------------------------------------------ tone sandhi (ToneSandhi)

    private static List<string> Chars(string word) => JiebaSegmenter.Codepoints(word);
    private static string WithTone(string final, char tone) => final.Substring(0, final.Length - 1) + tone;

    private List<(string Word, string Pos)> PreMerge(List<(string Word, string Pos)> seg)
    {
        seg = MergeBu(seg);
        try
        {
            seg = MergeYi(seg);
        }
        catch (ArgumentOutOfRangeException)
        {
            // "_merge_yi failed": the reference keeps the segmentation as it was.
        }
        seg = MergeReduplication(seg);
        seg = MergeContinuousThreeTones(seg);
        seg = MergeContinuousThreeTones2(seg);
        return MergeEr(seg);
    }

    private static List<(string, string)> MergeBu(List<(string Word, string Pos)> seg)
    {
        var result = new List<(string, string)>();
        string last = "";
        foreach (var (w, pos) in seg)
        {
            string word = last == "不" ? last + w : w;
            if (word != "不") result.Add((word, pos));
            last = word;
        }
        if (last == "不") result.Add((last, "d"));
        return result;
    }

    // _merge_yi; an index past the end raises, and PreMerge then keeps the input (as the reference's bare except does).
    private static List<(string, string)> MergeYi(List<(string Word, string Pos)> seg)
    {
        var merged = new List<(string Word, string Pos)>();
        for (int i = 0; i < seg.Count; i++)
        {
            var (word, pos) = seg[i];
            if (i - 1 >= 0 && word == "一" && i + 1 < seg.Count && seg[i - 1].Word == seg[i + 1].Word && seg[i - 1].Pos == "v")
            {
                if (i - 1 >= merged.Count) throw new ArgumentOutOfRangeException(nameof(seg));
                merged[i - 1] = (merged[i - 1].Word + "一" + merged[i - 1].Word, merged[i - 1].Pos);
            }
            else if (i - 2 >= 0 && seg[i - 1].Word == "一" && seg[i - 2].Word == word && pos == "v")
            {
                continue;
            }
            else
            {
                merged.Add((word, pos));
            }
        }
        var result = new List<(string Word, string Pos)>();
        foreach (var (word, pos) in merged)
        {
            if (result.Count > 0 && result[result.Count - 1].Word == "一")
                result[result.Count - 1] = (result[result.Count - 1].Word + word, result[result.Count - 1].Pos);
            else
                result.Add((word, pos));
        }
        return result;
    }

    private static List<(string, string)> MergeReduplication(List<(string Word, string Pos)> seg)
    {
        var result = new List<(string Word, string Pos)>();
        foreach (var (word, pos) in seg)
        {
            if (result.Count > 0 && word == result[result.Count - 1].Word)
                result[result.Count - 1] = (result[result.Count - 1].Word + word, result[result.Count - 1].Pos);
            else
                result.Add((word, pos));
        }
        return result;
    }

    private static bool AllToneThree(IEnumerable<string> finals) => finals.All(f => f.Length > 0 && f[f.Length - 1] == '3');
    private static bool IsReduplication(string word) { var c = Chars(word); return c.Count == 2 && c[0] == c[1]; }

    private List<(string, string)> MergeContinuousThreeTones(List<(string Word, string Pos)> seg)
    {
        var finals = seg.Select(s => Pypinyin050.Default.LazyPinyin(s.Word).Select(r => r.Final).ToList()).ToList();
        var result = new List<(string Word, string Pos)>();
        var mergeLast = new bool[seg.Count];
        for (int i = 0; i < seg.Count; i++)
        {
            var (word, pos) = seg[i];
            if (i - 1 >= 0 && AllToneThree(finals[i - 1]) && AllToneThree(finals[i]) && !mergeLast[i - 1]
                && !IsReduplication(seg[i - 1].Word) && Chars(seg[i - 1].Word).Count + Chars(word).Count <= 3)
            {
                result[result.Count - 1] = (result[result.Count - 1].Word + word, result[result.Count - 1].Pos);
                mergeLast[i] = true;
            }
            else
            {
                result.Add((word, pos));
            }
        }
        return result;
    }

    private List<(string, string)> MergeContinuousThreeTones2(List<(string Word, string Pos)> seg)
    {
        var finals = seg.Select(s => Pypinyin050.Default.LazyPinyin(s.Word).Select(r => r.Final).ToList()).ToList();
        var result = new List<(string Word, string Pos)>();
        var mergeLast = new bool[seg.Count];
        static char LastChar(string s) => s[s.Length - 1];
        for (int i = 0; i < seg.Count; i++)
        {
            var (word, pos) = seg[i];
            if (i - 1 >= 0 && LastChar(finals[i - 1][finals[i - 1].Count - 1]) == '3' && LastChar(finals[i][0]) == '3' && !mergeLast[i - 1]
                && !IsReduplication(seg[i - 1].Word) && Chars(seg[i - 1].Word).Count + Chars(word).Count <= 3)
            {
                result[result.Count - 1] = (result[result.Count - 1].Word + word, result[result.Count - 1].Pos);
                mergeLast[i] = true;
            }
            else
            {
                result.Add((word, pos));
            }
        }
        return result;
    }

    private static List<(string, string)> MergeEr(List<(string Word, string Pos)> seg)
    {
        var result = new List<(string Word, string Pos)>();
        for (int i = 0; i < seg.Count; i++)
        {
            var (word, pos) = seg[i];
            if (i - 1 >= 0 && word == "儿" && seg[i - 1].Word != "#")
                result[result.Count - 1] = (result[result.Count - 1].Word + word, result[result.Count - 1].Pos);
            else
                result.Add((word, pos));
        }
        return result;
    }

    private List<string> ModifiedTone(string word, string pos, List<string> finals)
    {
        finals = BuSandhi(word, finals);
        finals = YiSandhi(word, finals);
        finals = NeutralSandhi(word, pos, finals);
        return ThreeSandhi(word, finals);
    }

    private static List<string> BuSandhi(string word, List<string> finals)
    {
        var c = Chars(word);
        if (c.Count == 3 && c[1] == "不") finals[1] = WithTone(finals[1], '5');
        else
            for (int i = 0; i < c.Count; i++)
                if (c[i] == "不" && i + 1 < c.Count && finals[i + 1][finals[i + 1].Length - 1] == '4') finals[i] = WithTone(finals[i], '2');
        return finals;
    }

    private const string SandhiPunctuation = "：，；。？！“”‘’':,;.?!";

    private List<string> YiSandhi(string word, List<string> finals)
    {
        var c = Chars(word);
        if (c.Contains("一") && c.Where(x => x != "一").All(IsNumeric)) return finals;
        if (c.Count == 3 && c[1] == "一" && c[0] == c[2]) finals[1] = WithTone(finals[1], '5');
        else if (word.StartsWith("第一", StringComparison.Ordinal)) finals[1] = WithTone(finals[1], '1');
        else
            for (int i = 0; i < c.Count; i++)
            {
                if (c[i] != "一" || i + 1 >= c.Count) continue;
                if (finals[i + 1][finals[i + 1].Length - 1] == '4') finals[i] = WithTone(finals[i], '2');
                else if (!SandhiPunctuation.Contains(c[i + 1])) finals[i] = WithTone(finals[i], '4');
            }
        return finals;
    }

    private List<string> SplitWord(string word)
    {
        // jieba.cut_for_search, sorted by length (stably), the shortest taken as one part.
        var parts = JiebaSegmenter.Default.CutForSearch(word).Select((w, i) => (w, i)).OrderBy(p => Chars(p.w).Count).ThenBy(p => p.i).Select(p => p.w).ToList();
        string first = parts[0];
        int at = word.IndexOf(first, StringComparison.Ordinal);
        if (at == 0) return new List<string> { first, word.Substring(first.Length) };
        return new List<string> { word.Substring(0, word.Length - first.Length), first };
    }

    private List<string> NeutralSandhi(string word, string pos, List<string> finals)
    {
        var c = Chars(word);
        for (int j = 0; j < c.Count; j++)
            if (j - 1 >= 0 && c[j] == c[j - 1] && "nva".IndexOf(pos[0]) >= 0 && !_mustNotNeutral.Contains(word))
                finals[j] = WithTone(finals[j], '5');
        int ge = c.IndexOf("个");
        string last = c.Count > 0 ? c[c.Count - 1] : "";
        string lastTwo = c.Count >= 2 ? c[c.Count - 2] + c[c.Count - 1] : word;
        if (c.Count >= 1 && "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶".Contains(last)) finals[finals.Count - 1] = WithTone(finals[finals.Count - 1], '5');
        else if (c.Count >= 1 && "的地得".Contains(last)) finals[finals.Count - 1] = WithTone(finals[finals.Count - 1], '5');
        else if (c.Count > 1 && "们子".Contains(last) && (pos == "r" || pos == "n") && !_mustNotNeutral.Contains(word))
            finals[finals.Count - 1] = WithTone(finals[finals.Count - 1], '5');
        else if (c.Count > 1 && "上下里".Contains(last) && (pos == "s" || pos == "l" || pos == "f"))
            finals[finals.Count - 1] = WithTone(finals[finals.Count - 1], '5');
        else if (c.Count > 1 && "来去".Contains(last) && "上下进出回过起开".Contains(c[c.Count - 2]))
            finals[finals.Count - 1] = WithTone(finals[finals.Count - 1], '5');
        else if ((ge >= 1 && (IsNumeric(c[ge - 1]) || "几有两半多各整每做是".Contains(c[ge - 1]))) || word == "个")
            finals[ge] = WithTone(finals[ge], '5');
        else if (_mustNeutral.Contains(word) || _mustNeutral.Contains(lastTwo))
            finals[finals.Count - 1] = WithTone(finals[finals.Count - 1], '5');

        var words = SplitWord(word);
        int firstLength = Chars(words[0]).Count;
        var parts = new List<List<string>> { finals.Take(firstLength).ToList(), finals.Skip(firstLength).ToList() };
        for (int i = 0; i < words.Count; i++)
        {
            var w = Chars(words[i]);
            string tail = w.Count >= 2 ? w[w.Count - 2] + w[w.Count - 1] : words[i];
            if (_mustNeutral.Contains(words[i]) || _mustNeutral.Contains(tail))
            {
                // finals_list[i][-1] on an empty part raises in the reference.
                if (parts[i].Count == 0) throw new ArgumentException($"The word {word} splits into an empty part.");
                parts[i][parts[i].Count - 1] = WithTone(parts[i][parts[i].Count - 1], '5');
            }
        }
        return parts[0].Concat(parts[1]).ToList();
    }

    private List<string> ThreeSandhi(string word, List<string> finals)
    {
        int length = Chars(word).Count;
        if (length == 2 && AllToneThree(finals))
        {
            finals[0] = WithTone(finals[0], '2');
        }
        else if (length == 3)
        {
            var words = SplitWord(word);
            int firstLength = Chars(words[0]).Count;
            if (AllToneThree(finals))
            {
                if (firstLength == 2)
                {
                    finals[0] = WithTone(finals[0], '2');
                    finals[1] = WithTone(finals[1], '2');
                }
                else if (firstLength == 1)
                {
                    finals[1] = WithTone(finals[1], '2');
                }
            }
            else
            {
                var parts = new List<List<string>> { finals.Take(firstLength).ToList(), finals.Skip(firstLength).ToList() };
                for (int i = 0; i < parts.Count; i++)
                {
                    var sub = parts[i];
                    if (AllToneThree(sub) && sub.Count == 2) sub[0] = WithTone(sub[0], '2');
                    else if (i == 1 && !AllToneThree(sub) && sub[0][sub[0].Length - 1] == '3'
                             && parts[0][parts[0].Count - 1][parts[0][parts[0].Count - 1].Length - 1] == '3')
                        parts[0][parts[0].Count - 1] = WithTone(parts[0][parts[0].Count - 1], '2');
                    finals = parts[0].Concat(parts[1]).ToList();
                }
            }
        }
        else if (length == 4)
        {
            var parts = new List<List<string>> { finals.Take(2).ToList(), finals.Skip(2).ToList() };
            finals = new List<string>();
            foreach (var sub in parts)
            {
                if (AllToneThree(sub)) sub[0] = WithTone(sub[0], '2');
                finals.AddRange(sub);
            }
        }
        return finals;
    }
}
