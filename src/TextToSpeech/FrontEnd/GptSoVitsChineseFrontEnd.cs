using System.Globalization;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// GPT-SoVITS's Chinese front end (RVC-Boss/GPT-SoVITS at b4570b8, MIT): <c>text/chinese.py</c> (version v1: pypinyin's
/// readings) and <c>text/chinese2.py</c> (v2 and later: G2PW's readings corrected by the polyphonic dictionary, then
/// erhua), over its <c>text_normalize</c> (zh_normalization, the punctuation map, everything outside Chinese and the six
/// punctuation marks dropped, repeated punctuation collapsed) and its tone sandhi (<c>text/tone_sandhi.py</c>), with
/// jieba_fast's part-of-speech segmentation.
/// </summary>
internal sealed class GptSoVitsChineseFrontEnd
{
    private static readonly Lazy<GptSoVitsChineseFrontEnd> SharedInstance = new(() => new GptSoVitsChineseFrontEnd(), isThreadSafe: true);

    public static GptSoVitsChineseFrontEnd Default => SharedInstance.Value;

    /// <summary>text/symbols.py's punctuation.</summary>
    internal static readonly string[] Punctuation = { "!", "?", "…", ",", ".", "-" };

    private readonly GptSoVitsChineseData _data = GptSoVitsChineseData.Instance;
    private readonly GptSoVitsPinyin _pinyin = GptSoVitsPinyin.Default;

    private GptSoVitsChineseFrontEnd()
    {
    }

    // ------------------------------------------------------------------ text_normalize

    private static readonly (string From, string To)[] Replacements =
    {
        ("：", ","), ("；", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", "."), ("·", ","),
        ("、", ","), ("...", "…"), ("$", "."), ("/", ","), ("—", "-"), ("~", "…"), ("～", "…"),
    };

    private static readonly Regex ReplacementPattern = new(string.Join("|", Replacements.Select(r => Regex.Escape(r.From))));
    private static readonly Regex Outside = new("[^一-龥!?…,.\\-]+");
    private static readonly Regex RepeatedPunctuation = new("([!?…,.\\-])([!?…,.\\-])+");

    /// <summary><c>replace_punctuation</c>.</summary>
    internal static string ReplacePunctuation(string text)
    {
        text = text.Replace("嗯", "恩").Replace("呣", "母");
        text = ReplacementPattern.Replace(text, m => Replacements.First(r => r.From == m.Value).To);
        return Outside.Replace(text, "");
    }

    /// <summary><c>text_normalize</c> (identical in chinese.py and chinese2.py).</summary>
    public string TextNormalize(string text)
    {
        string dest = string.Concat(GptSoVitsChineseNormalizer.Normalize(text).Select(ReplacePunctuation));
        return RepeatedPunctuation.Replace(dest, "$1");
    }

    // ------------------------------------------------------------------ g2p

    private static readonly Regex SentenceSplit = new("(?<=[!?…,.\\-])\\s*");
    private static readonly Regex Latin = new("[a-zA-Z]+");

    /// <summary>
    /// <c>g2p(text)</c> of normalized text: the phones and the phones per character. Version v1 reads pypinyin; later
    /// versions read <paramref name="g2pw"/> (G2PW's per-character readings of each sentence; see
    /// <see cref="G2pwConverter{T}"/>).
    /// </summary>
    public (List<string> Phones, List<int> WordToPhones) G2P(string text, bool v1, Func<List<string>, List<List<string?>>>? g2pw = null)
    {
        if (!v1 && g2pw is null) throw new ArgumentNullException(nameof(g2pw), "Versions after v1 read G2PW.");
        var segments = SentenceSplit.Split(text).Where(s => PythonText.Strip(s).Length > 0).Select(s => Latin.Replace(s, "")).ToList();
        var phones = new List<string>();
        var wordToPhones = new List<int>();
        List<List<string?>> predictions = new();
        int cursor = 0;
        if (!v1)
        {
            var batch = segments.Where(s => s.Length > 0).ToList();
            if (batch.Count > 0) predictions = g2pw!(batch);
        }
        foreach (var seg in segments)
        {
            var cut = PreMerge(Jieba.Default.PosCut(seg));
            var initials = new List<string>();
            var finals = new List<string>();
            if (v1)
            {
                foreach (var (word, pos) in cut)
                {
                    if (pos == "eng") continue;
                    var readings = _pinyin.Read(word);
                    initials.AddRange(readings.Select(r => r.Initial));
                    finals.AddRange(ModifiedTone(word, pos, readings.Select(r => r.Final).ToList()));
                }
            }
            else
            {
                List<string?> pinyins = new();
                if (seg.Length > 0) pinyins = predictions[cursor++];
                int previous = 0;
                foreach (var (word, pos) in cut)
                {
                    int length = ChineseText.Codepoints(word).Count;
                    int now = previous + length;
                    if (pos == "eng")
                    {
                        previous = now;
                        continue;
                    }
                    var wordPinyins = pinyins.Skip(previous).Take(now - previous).ToList();
                    wordPinyins = CorrectPronunciation(word, wordPinyins);
                    var subInitials = new List<string>();
                    var subFinals = new List<string>();
                    foreach (var pinyin in wordPinyins)
                    {
                        if (pinyin is null) throw new InvalidOperationException($"G2PW gave no reading in '{seg}'.");
                        if (pinyin.Length == 0) throw new IndexOutOfRangeException($"An empty reading in '{seg}'.");
                        if (IsAlpha(pinyin))
                        {
                            subInitials.Add(GptSoVitsPinyin.ToInitials(pinyin));
                            subFinals.Add(GptSoVitsPinyin.ToFinalsTone3(pinyin, neutralToneWithFive: true));
                        }
                        else
                        {
                            subInitials.Add(pinyin);
                            subFinals.Add(pinyin);
                        }
                    }
                    previous = now;
                    subFinals = ModifiedTone(word, pos, subFinals);
                    (subInitials, subFinals) = MergeErhua(subInitials, subFinals, word, pos);
                    initials.AddRange(subInitials);
                    finals.AddRange(subFinals);
                }
            }
            AppendPhones(seg, initials, finals, phones, wordToPhones);
        }
        return (phones, wordToPhones);
    }

    // pinyin[0].isalpha()
    private static bool IsAlpha(string text)
    {
        var category = CharUnicodeInfo.GetUnicodeCategory(text, 0);
        return category is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
            or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter;
    }

    // g2pw.correct_pronunciation: the word's corrected readings, else each character's first corrected reading.
    private List<string?> CorrectPronunciation(string word, List<string?> wordPinyins)
    {
        if (_data.Corrections.TryGetValue(word, out var corrected)) return corrected.Cast<string?>().ToList();
        var chars = ChineseText.Codepoints(word);
        for (int i = 0; i < chars.Count; i++)
        {
            if (!_data.Corrections.TryGetValue(chars[i], out var readings)) continue;
            if (i >= wordPinyins.Count) throw new ArgumentOutOfRangeException(nameof(wordPinyins), $"'{word}' has more characters than readings.");
            wordPinyins[i] = readings[0];
        }
        return wordPinyins;
    }

    // chinese2._merge_erhua
    private (List<string>, List<string>) MergeErhua(List<string> initials, List<string> finals, string word, string pos)
    {
        var chars = ChineseText.Codepoints(word);
        for (int i = 0; i < finals.Count; i++)
            if (i == finals.Count - 1 && chars[i] == "儿" && finals[i] == "er1") finals[i] = "er2";
        if (!_data.MustErhua.Contains(word) && (_data.NotErhua.Contains(word) || pos == "a" || pos == "j" || pos == "nr"))
            return (initials, finals);
        if (finals.Count != chars.Count) return (initials, finals);
        string lastTwo = chars.Count >= 2 ? chars[chars.Count - 2] + chars[chars.Count - 1] : word;
        var newInitials = new List<string>();
        var newFinals = new List<string>();
        for (int i = 0; i < finals.Count; i++)
        {
            string phn = finals[i];
            if (i == finals.Count - 1 && chars[i] == "儿" && (phn == "er2" || phn == "er5") && !_data.NotErhua.Contains(lastTwo) && newFinals.Count > 0)
            {
                string previous = newFinals[newFinals.Count - 1];
                phn = "er" + previous[previous.Length - 1];
            }
            newInitials.Add(initials[i]);
            newFinals.Add(phn);
        }
        return (newInitials, newFinals);
    }

    // The shared tail of _g2p: each (initial, final) to its opencpop phones.
    private void AppendPhones(string seg, List<string> initials, List<string> finals, List<string> phones, List<int> wordToPhones)
    {
        for (int i = 0; i < Math.Min(initials.Count, finals.Count); i++)
        {
            string c = initials[i], v = finals[i];
            if (c == v)
            {
                if (!Punctuation.Contains(c)) throw new ArgumentException($"'{c}' has no Chinese reading and is not GPT-SoVITS punctuation.", nameof(seg));
                phones.Add(c);
                wordToPhones.Add(1);
                continue;
            }
            if (v.Length == 0) throw new ArgumentException($"The reading {c} of '{seg}' has no final.", nameof(seg));
            string withoutTone = v.Substring(0, v.Length - 1);
            char tone = v[v.Length - 1];
            string pinyin = c + withoutTone;
            if ("12345".IndexOf(tone) < 0) throw new ArgumentException($"The reading {c}{v} has no tone.", nameof(seg));
            if (c.Length > 0)
            {
                string? replaced = withoutTone switch { "uei" => "ui", "iou" => "iu", "uen" => "un", _ => null };
                if (replaced is not null) pinyin = c + replaced;
            }
            else
            {
                string? whole = pinyin switch { "ing" => "ying", "i" => "yi", "in" => "yin", "u" => "wu", _ => null };
                if (whole is not null) pinyin = whole;
                else
                {
                    if (pinyin.Length == 0) throw new ArgumentException($"The reading {v} of '{seg}' is empty.", nameof(seg));
                    string? first = pinyin[0] switch { 'v' => "yu", 'e' => "e", 'i' => "y", 'u' => "w", _ => null };
                    if (first is not null) pinyin = first + pinyin.Substring(1);
                }
            }
            if (!_data.Opencpop.TryGetValue(pinyin, out var symbols))
                throw new ArgumentException($"The syllable {pinyin} ({c}{v} in '{seg}') is not in GPT-SoVITS's phone table.", nameof(seg));
            var pair = symbols.Split(' ');
            if (pair.Length != 2) throw new InvalidDataException($"The phone table maps {pinyin} to '{symbols}', not two phones.");
            phones.Add(pair[0]);
            phones.Add(pair[1] + tone);
            wordToPhones.Add(2);
        }
    }

    // ------------------------------------------------------------------ tone sandhi (text/tone_sandhi.py)

    private static List<string> Chars(string word) => ChineseText.Codepoints(word);

    // finals[j][:-1] + tone
    private static string WithTone(string final, char tone) => (final.Length == 0 ? "" : final.Substring(0, final.Length - 1)) + tone;

    private static char Last(string s) => s[s.Length - 1];

    private bool IsNumeric(string c) => _data.Numeric.Contains(c);

    /// <summary><c>pre_merge_for_modify</c>.</summary>
    internal List<(string Word, string Pos)> PreMerge(List<(string Word, string Pos)> seg)
    {
        seg = MergeBu(seg);
        seg = MergeYi(seg);
        seg = MergeReduplication(seg);
        try
        {
            seg = MergeContinuousThreeTones(seg);
        }
        catch (Exception e) when (e is IndexOutOfRangeException or ArgumentOutOfRangeException)
        {
            // "_merge_continuous_three_tones failed": the segmentation stays as it was.
        }
        try
        {
            seg = MergeContinuousThreeTones2(seg);
        }
        catch (Exception e) when (e is IndexOutOfRangeException or ArgumentOutOfRangeException)
        {
            // "_merge_continuous_three_tones_2 failed".
        }
        return MergeEr(seg);
    }

    private static List<(string Word, string Pos)> MergeBu(List<(string Word, string Pos)> seg)
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

    // _merge_yi: "v 一 v" (the left word as merged so far) into one word, then a lone "一" onto the word after it.
    private static List<(string Word, string Pos)> MergeYi(List<(string Word, string Pos)> seg)
    {
        var merged = new List<(string Word, string Pos)>();
        int i = 0;
        while (i < seg.Count)
        {
            var (word, pos) = seg[i];
            if (i - 1 >= 0 && word == "一" && i + 1 < seg.Count)
            {
                var last = merged.Count > 0 ? merged[merged.Count - 1] : seg[i - 1];
                if (last.Word == seg[i + 1].Word && last.Pos == "v" && seg[i + 1].Pos == "v")
                {
                    merged[merged.Count - 1] = (last.Word + "一" + seg[i + 1].Word, last.Pos);
                    i += 2;
                    continue;
                }
            }
            merged.Add((word, pos));
            i++;
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

    private static List<(string Word, string Pos)> MergeReduplication(List<(string Word, string Pos)> seg)
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

    // x[-1] == "3" for every final (an empty final raises, as in the reference).
    private static bool AllToneThree(IEnumerable<string> finals) => finals.All(f => Last(f) == '3');

    private static bool IsReduplication(string word) { var c = Chars(word); return c.Count == 2 && c[0] == c[1]; }

    private List<List<string>> FinalsOf(List<(string Word, string Pos)> seg)
        => seg.Select(s => _pinyin.Read(s.Word).Select(r => r.Final).ToList()).ToList();

    private List<(string Word, string Pos)> MergeContinuousThreeTones(List<(string Word, string Pos)> seg)
    {
        var finals = FinalsOf(seg);
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

    private List<(string Word, string Pos)> MergeContinuousThreeTones2(List<(string Word, string Pos)> seg)
    {
        var finals = FinalsOf(seg);
        var result = new List<(string Word, string Pos)>();
        var mergeLast = new bool[seg.Count];
        for (int i = 0; i < seg.Count; i++)
        {
            var (word, pos) = seg[i];
            if (i - 1 >= 0 && Last(finals[i - 1][finals[i - 1].Count - 1]) == '3' && Last(finals[i][0]) == '3' && !mergeLast[i - 1]
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

    private static List<(string Word, string Pos)> MergeEr(List<(string Word, string Pos)> seg)
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

    /// <summary><c>modified_tone</c>.</summary>
    internal List<string> ModifiedTone(string word, string pos, List<string> finals)
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
                if (c[i] == "不" && i + 1 < c.Count && Last(finals[i + 1]) == '4') finals[i] = WithTone(finals[i], '2');
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
                if (Last(finals[i + 1]) == '4') finals[i] = WithTone(finals[i], '2');
                else if (!SandhiPunctuation.Contains(c[i + 1])) finals[i] = WithTone(finals[i], '4');
            }
        return finals;
    }

    // _split_word over jieba_fast's cut_for_search.
    private static List<string> SplitWord(string word)
    {
        var parts = Jieba.Default.CutForSearch(word, jiebaFast: true).Select((w, i) => (w, i))
            .OrderBy(p => Chars(p.w).Count).ThenBy(p => p.i).Select(p => p.w).ToList();
        string first = parts[0];
        int at = word.IndexOf(first, StringComparison.Ordinal);
        if (at == 0) return new List<string> { first, word.Substring(first.Length) };
        return new List<string> { word.Substring(0, word.Length - first.Length), first };
    }

    private const string NeutralParticles = "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶";

    private List<string> NeutralSandhi(string word, string pos, List<string> finals)
    {
        var c = Chars(word);
        for (int j = 0; j < c.Count; j++)
            if (j - 1 >= 0 && c[j] == c[j - 1] && "nva".IndexOf(pos[0]) >= 0 && !_data.MustNotNeutral.Contains(word))
                finals[j] = WithTone(finals[j], '5');
        int ge = c.IndexOf("个");
        string last = c.Count > 0 ? c[c.Count - 1] : "";
        string lastTwo = c.Count >= 2 ? c[c.Count - 2] + c[c.Count - 1] : word;
        int final = finals.Count - 1;
        if (c.Count >= 1 && NeutralParticles.Contains(last)) finals[final] = WithTone(finals[final], '5');
        else if (c.Count >= 1 && "的地得".Contains(last)) finals[final] = WithTone(finals[final], '5');
        else if (c.Count == 1 && "了着过".Contains(word) && (pos == "ul" || pos == "uz" || pos == "ug"))
            finals[final] = WithTone(finals[final], '5');
        else if (c.Count > 1 && "们子".Contains(last) && (pos == "r" || pos == "n") && !_data.MustNotNeutral.Contains(word))
            finals[final] = WithTone(finals[final], '5');
        else if (c.Count > 1 && "上下里".Contains(last) && (pos == "s" || pos == "l" || pos == "f"))
            finals[final] = WithTone(finals[final], '5');
        else if (c.Count > 1 && "来去".Contains(last) && "上下进出回过起开".Contains(c[c.Count - 2]))
            finals[final] = WithTone(finals[final], '5');
        else if ((ge >= 1 && (IsNumeric(c[ge - 1]) || "几有两半多各整每做是".Contains(c[ge - 1]))) || word == "个")
            finals[ge] = WithTone(finals[ge], '5');
        else if (_data.MustNeutral.Contains(word) || _data.MustNeutral.Contains(lastTwo))
            finals[final] = WithTone(finals[final], '5');

        var words = SplitWord(word);
        int firstLength = Chars(words[0]).Count;
        var parts = new List<List<string>> { finals.Take(firstLength).ToList(), finals.Skip(firstLength).ToList() };
        for (int i = 0; i < words.Count; i++)
        {
            var w = Chars(words[i]);
            string tail = w.Count >= 2 ? w[w.Count - 2] + w[w.Count - 1] : words[i];
            if (_data.MustNeutral.Contains(words[i]) || _data.MustNeutral.Contains(tail))
            {
                if (parts[i].Count == 0) throw new ArgumentException($"The word {word} splits into an empty part.");
                parts[i][parts[i].Count - 1] = WithTone(parts[i][parts[i].Count - 1], '5');
            }
        }
        return parts[0].Concat(parts[1]).ToList();
    }

    private static List<string> ThreeSandhi(string word, List<string> finals)
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
                    else if (i == 1 && !AllToneThree(sub) && Last(sub[0]) == '3' && Last(parts[0][parts[0].Count - 1]) == '3')
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
