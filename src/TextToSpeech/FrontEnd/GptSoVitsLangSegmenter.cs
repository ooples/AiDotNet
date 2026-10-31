using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// BudouX 0.9.3 (Google; Apache-2.0): a sentence cut into chunks where the model's unigram, bigram and trigram scores
/// around each boundary sum above zero (<c>Parser.parse</c>), with its Japanese and Simplified Chinese models.
/// </summary>
internal sealed class Budoux
{
    private static readonly Lazy<Budoux> JapaneseInstance = new(() => new Budoux("budoux_ja.json.gz"), isThreadSafe: true);
    private static readonly Lazy<Budoux> ChineseInstance = new(() => new Budoux("budoux_zh_hans.json.gz"), isThreadSafe: true);
    private readonly Dictionary<string, Dictionary<string, int>> _model = new(StringComparer.Ordinal);
    private readonly double _baseScore;

    public static Budoux Japanese => JapaneseInstance.Value;

    public static Budoux SimplifiedChinese => ChineseInstance.Value;

    private Budoux(string resource)
    {
        var json = JObject.Parse(string.Join("\n", ChineseText.ReadLines(resource)));
        long total = 0;
        foreach (var group in json)
        {
            var scores = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var entry in (JObject)group.Value!)
            {
                scores[entry.Key] = (int)entry.Value!;
                total += scores[entry.Key];
            }
            _model[group.Key] = scores;
        }
        _baseScore = -total * 0.5;
    }

    private int Score(string group, List<string> s, int from, int to)
        => _model.TryGetValue(group, out var g) && g.TryGetValue(string.Concat(s.GetRange(from, to - from)), out int v) ? v : 0;

    /// <summary><c>parser.parse(sentence)</c>.</summary>
    public List<string> Parse(string sentence)
    {
        var s = ChineseText.Codepoints(sentence);
        var chunks = new List<string>();
        if (s.Count == 0) return chunks;
        chunks.Add(s[0]);
        for (int i = 1; i < s.Count; i++)
        {
            double score = _baseScore;
            if (i > 2) score += Score("UW1", s, i - 3, i - 2);
            if (i > 1) score += Score("UW2", s, i - 2, i - 1);
            score += Score("UW3", s, i - 1, i);
            score += Score("UW4", s, i, i + 1);
            if (i + 1 < s.Count) score += Score("UW5", s, i + 1, i + 2);
            if (i + 2 < s.Count) score += Score("UW6", s, i + 2, i + 3);
            if (i > 1) score += Score("BW1", s, i - 2, i);
            score += Score("BW2", s, i - 1, i + 1);
            if (i + 1 < s.Count) score += Score("BW3", s, i, i + 2);
            if (i > 2) score += Score("TW1", s, i - 3, i);
            if (i > 1) score += Score("TW2", s, i - 2, i + 1);
            if (i + 1 < s.Count) score += Score("TW3", s, i - 1, i + 2);
            if (i + 2 < s.Count) score += Score("TW4", s, i, i + 3);
            if (score > 0) chunks.Add(s[i]);
            else chunks[chunks.Count - 1] += s[i];
        }
        return chunks;
    }
}

/// <summary>
/// split-lang 2.1.1 (DoodleBears; MIT): <c>LangSplitter.split_by_lang</c>, which cuts text into Chinese/Japanese, Korean,
/// digit, punctuation, newline and other sections, cuts the Chinese/Japanese sections with BudouX and the others at
/// spaces, labels each piece with fast_langdetect, then merges pieces and sections by its rules.
/// </summary>
/// <remarks>Lengths and indices count code points, as Python's do. Its pieces are shared objects that later merges
/// change in place, as the reference's pydantic models are.</remarks>
internal sealed class SplitLang
{
    internal sealed class SubString
    {
        public string Lang = "", Text = "";
        public int Index, Length;
    }

    private enum SectionType { ZhJa, Ko, Punctuation, Newline, Digit, Others }

    private sealed class Section
    {
        public SectionType Type;
        public string Text = "";
        public List<SubString> Substrings = new();
    }

    private const string Punctuation =
        "\u301c~,.;:!?\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\u00b7([{<\uff08\u3010\u300a\u3008\u300c\u300e\u201c\u2018)]}>\uff09\u3011\u300b\u3009\u300d\u300f\u201d\u2019\"-_\u2014\u2014#$%&\u2026\u2026\uffe5'*+<=>?@[\\]^_`{|}~";

    private static readonly Dictionary<string, string> ZhJaLangMap = new(StringComparer.Ordinal)
    {
        ["zh"] = "zh", ["yue"] = "zh", ["wuu"] = "zh", ["zh-cn"] = "zh", ["zh-tw"] = "x", ["ja"] = "ja",
    };

    private readonly FastLangDetect _detector;
    private readonly Dictionary<string, string> _langMap;
    private readonly bool _mergeAcrossDigit, _mergeAcrossPunctuation, _mergeAcrossNewline;
    private readonly string _notMergePunctuation;
    private const string DefaultLang = "x";

    public SplitLang(FastLangDetect detector, Dictionary<string, string> langMap, bool mergeAcrossDigit = true,
        bool mergeAcrossPunctuation = true, bool mergeAcrossNewline = true, string notMergePunctuation = "")
    {
        _detector = detector;
        _langMap = langMap;
        _mergeAcrossDigit = mergeAcrossDigit;
        _mergeAcrossPunctuation = mergeAcrossPunctuation;
        _mergeAcrossNewline = mergeAcrossNewline;
        _notMergePunctuation = notMergePunctuation;
    }

    private static readonly Regex Hangul = new("[\uac00-\ud7af]");
    private static readonly Regex Kana = new("[\u3040-\u30ff\u3005]");
    private static readonly Regex ZhJa = new("[\u4e00-\u9fff\u3040-\u30ff\u3005]");

    private static bool ContainsKana(string text) => Kana.IsMatch(text);

    private static int Length(string text) => ChineseText.Codepoints(text).Count;

    private string FastLangDetectTop(string text) => _detector.Detect(text)[0].Lang.ToLowerInvariant();

    private string DetectCombined(string text, SectionType type)
        => type == SectionType.ZhJa && ContainsKana(text) ? "ja" : FastLangDetectTop(text);

    private List<string> PossibleDetections(string text)
        => _detector.Detect(PythonText.Strip(text.Replace("\n", "")), k: 5, threshold: 0.01f).Select(r => r.Lang).ToList();

    /// <summary><c>split_by_lang(text)</c>.</summary>
    public List<SubString> SplitByLang(string text)
    {
        var sections = Split(PreSplit(text));
        if (_mergeAcrossDigit) sections = MergeAcross(sections, SectionType.Digit);
        if (_mergeAcrossPunctuation) sections = MergeAcross(sections, SectionType.Punctuation);
        if (_mergeAcrossNewline) sections = MergeAcross(sections, SectionType.Newline);
        foreach (var section in sections)
            if (section.Type == SectionType.ZhJa) section.Substrings = SpecialMergeForZhJa(section.Substrings);
        return sections.SelectMany(s => s.Substrings).ToList();
    }

    private List<Section> PreSplit(string text)
    {
        var chars = ChineseText.Codepoints(PythonText.Strip(text));
        var sections = new List<Section>();
        var current = SectionType.Others;
        var buffer = new List<string>();
        void Add(SectionType type)
        {
            if (buffer.Count == 0) return;
            sections.Add(new Section { Type = type, Text = string.Concat(buffer) });
            buffer.Clear();
        }
        for (int index = 0; index < chars.Count; index++)
        {
            string c = chars[index];
            if (ZhJa.IsMatch(c))
            {
                if (current != SectionType.ZhJa) { Add(current); current = SectionType.ZhJa; }
            }
            else if (Hangul.IsMatch(c))
            {
                if (current != SectionType.Ko) { Add(current); current = SectionType.Ko; }
            }
            else if (PythonText.IsDigit(c))
            {
                if (current != SectionType.Digit) { Add(current); current = SectionType.Digit; }
            }
            else if (Punctuation.Contains(c))
            {
                if (!(c == "'" && index > 0 && !IsSpace(chars[index - 1])))
                {
                    Add(current);
                    current = SectionType.Punctuation;
                }
            }
            else if (IsSpace(c))
            {
                if (c is "\n" or "\r")
                {
                    Add(current);
                    current = SectionType.Newline;
                }
            }
            else if (current != SectionType.Others)
            {
                Add(current);
                current = SectionType.Others;
            }
            buffer.Add(c);
        }
        Add(current);
        return sections;
    }

    private static bool IsSpace(string c) => c.Length == 1 && PythonText.IsSpace(c[0]);

    private List<Section> Split(List<Section> sections)
    {
        int sectionIndex = 0;
        foreach (var section in sections)
        {
            int sectionLength = Length(section.Text);
            if (section.Type is SectionType.Punctuation or SectionType.Digit or SectionType.Newline)
            {
                string lang = section.Type switch { SectionType.Punctuation => "punctuation", SectionType.Digit => "digit", _ => "newline" };
                section.Substrings.Add(new SubString { Text = section.Text, Lang = lang, Index = sectionIndex, Length = sectionLength });
            }
            else
            {
                List<SubString> pieces;
                if (section.Type == SectionType.ZhJa) pieces = InitLangs(ParseZhJa(section.Text), SectionType.ZhJa, ZhJaLangMap);
                else if (section.Type == SectionType.Ko)
                    pieces = new List<SubString> { new() { Text = section.Text, Lang = "ko", Index = sectionIndex, Length = sectionLength } };
                else pieces = InitLangs(ParseWithoutZhJa(section.Text), SectionType.Others, null);
                foreach (var piece in pieces) piece.Index += sectionIndex;
                section.Substrings = pieces;
            }
            sectionIndex += sectionLength;
        }
        foreach (var section in sections)
        {
            if (section.Type is SectionType.Punctuation or SectionType.Newline) continue;
            section.Substrings = SmartMerge(section.Substrings, section.Type);
        }
        return sections;
    }

    private static List<string> ParseWithoutZhJa(string text)
    {
        var words = new List<string>();
        bool space = false;
        var chars = new List<string>();
        foreach (var c in ChineseText.Codepoints(text))
        {
            if (!IsSpace(c))
            {
                if (space)
                {
                    words.Add(string.Concat(chars));
                    chars.Clear();
                    space = false;
                }
                chars.Add(c);
            }
            else
            {
                space = true;
                chars.Add(c);
            }
        }
        if (chars.Count > 0) words.Add(string.Concat(chars));
        return words;
    }

    private static List<string> ParseZhJa(string text)
    {
        var pieces = Budoux.Japanese.Parse(text).SelectMany(Budoux.SimplifiedChinese.Parse).ToList();
        var merged = new List<string> { pieces[0] };
        foreach (var piece in pieces.Skip(1))
        {
            bool leftKana = ContainsKana(merged[merged.Count - 1]), currentKana = ContainsKana(piece);
            bool same = leftKana == currentKana;
            if (leftKana && same) merged[merged.Count - 1] += piece;
            else if (!leftKana && same && Length(piece) == 1) merged[merged.Count - 1] += piece;
            else merged.Add(piece);
        }
        if (merged.Count >= 2 && Length(merged[0]) == 1)
        {
            merged[1] = merged[0] + merged[1];
            merged.RemoveAt(0);
        }
        return merged;
    }

    private List<SubString> InitLangs(List<string> texts, SectionType type, Dictionary<string, string>? langMap)
    {
        langMap ??= _langMap;
        var result = new List<SubString>();
        int index = 0;
        foreach (var text in texts)
        {
            int length = Length(text);
            string lang = DetectCombined(text, type);
            result.Add(new SubString { Lang = langMap.TryGetValue(lang, out var mapped) ? mapped : DefaultLang, Text = text, Length = length, Index = index });
            index += length;
        }
        return result;
    }

    private List<SubString> SmartMerge(List<SubString> list, SectionType type)
    {
        list = MergeSubstrings(list);
        list = MergeMiddleToTwoSides(list);
        list = MergeSubstrings(list);
        list = MergeMiddleToTwoSides(list);     // _get_languages before it only detects; its result is discarded
        list = MergeSubstrings(list);
        list = FillUnknownLanguage(list);
        list = MergeSideToNear(list);
        list = MergeSubstrings(list);
        list = MergeMiddleToTwoSides(list);
        return MergeSubstrings(list);
    }

    private bool IsMergeMiddleToTwoSides(SubString left, SubString middle, SubString right)
    {
        bool noKana = !ContainsKana(middle.Text);
        bool possibleSame = PossibleDetections(middle.Text).Contains(left.Lang) && noKana;
        bool shortBetweenLong = middle.Length <= 4 && left.Length + right.Length >= 6 && noKana;
        return possibleSame || middle.Lang == "x" || shortBetweenLong;
    }

    private bool IsCurrentShortAndNearLong(SubString current, SubString near)
    {
        bool bothNoKana = !ContainsKana(current.Text) && !ContainsKana(near.Text);
        bool nearLongZh = near.Length > current.Length && near.Length >= 6 && near.Lang == "zh";
        bool possibleSame = PossibleDetections(current.Text).Contains(near.Lang) && bothNoKana;
        return (nearLongZh && bothNoKana) || possibleSame;
    }

    private List<SubString> MergeMiddleToTwoSides(List<SubString> substrings)
    {
        if (substrings.Count <= 2) return substrings;
        for (int index = 0; index < substrings.Count - 2; index++)
        {
            var left = substrings[index];
            var middle = substrings[index + 1];
            var right = substrings[index + 2];
            if (left.Lang == right.Lang && left.Lang != "x" && left.Lang != "newline" && IsMergeMiddleToTwoSides(left, middle, right))
                substrings[index + 1].Lang = left.Lang;
        }
        return substrings;
    }

    private List<SubString> MergeSideToNear(List<SubString> substrings)
    {
        bool isX = substrings[0].Lang == "x";
        bool shortNearLong = false, possibleSame = false;
        if (substrings.Count >= 2)
        {
            shortNearLong = IsCurrentShortAndNearLong(substrings[0], substrings[1]);
            possibleSame = PossibleDetections(substrings[0].Text).Contains(substrings[1].Lang) && substrings[1].Length <= 5;
        }
        if (isX || shortNearLong || possibleSame) substrings[0].Lang = NearestLang(substrings, 0, searchLeft: false);
        int last = substrings.Count - 1;
        isX = substrings[last].Lang == "x";
        shortNearLong = false;
        possibleSame = false;
        if (substrings.Count >= 2)
        {
            shortNearLong = IsCurrentShortAndNearLong(substrings[last], substrings[last - 1]);
            possibleSame = PossibleDetections(substrings[last].Text).Contains(substrings[last - 1].Lang) && substrings[last - 1].Length <= 5;
        }
        if (isX || shortNearLong || possibleSame) substrings[last].Lang = NearestLang(substrings, last, searchLeft: true);
        return substrings;
    }

    private static List<SubString> FillUnknownLanguage(List<SubString> substrings)
    {
        for (int index = 0; index < substrings.Count; index++)
        {
            if (substrings[index].Lang != "x") continue;
            if (index == 0) substrings[index].Lang = NearestLang(substrings, index, searchLeft: false);
            else if (index == substrings.Count - 1) substrings[index].Lang = NearestLang(substrings, index, searchLeft: true);
            else
            {
                bool left = Length(substrings[index - 1].Text) >= Length(substrings[index + 1].Text);
                substrings[index].Lang = NearestLang(substrings, index, left);
            }
        }
        return substrings;
    }

    private static string NearestLang(List<SubString> substrings, int index, bool searchLeft)
    {
        for (int i = 1; i < substrings.Count; i++)
        {
            int j = searchLeft ? index - i : index + i;
            if (j >= 0 && j < substrings.Count && substrings[j].Lang != "x" && substrings[j].Lang != "digit") return substrings[j].Lang;
        }
        return substrings[index].Lang;
    }

    private static List<SubString> MergeSubstrings(List<SubString> substrings)
    {
        var result = new List<SubString>();
        string lang = "";
        foreach (var block in substrings)
        {
            if (block.Lang != lang) result.Add(block);
            else
            {
                result[result.Count - 1].Text += block.Text;
                result[result.Count - 1].Length += block.Length;
            }
            lang = block.Lang;
        }
        return result;
    }

    private static List<SubString> SpecialMergeForZhJa(List<SubString> substrings)
    {
        if (substrings.Count == 1) return substrings;
        var counts = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["zh"] = 0, ["ja"] = 0, ["x"] = 0, ["digit"] = 0, ["punctuation"] = 0, ["newline"] = 0,
        };
        var result = new List<SubString>();
        int index = 0;
        while (index < substrings.Count)
        {
            var current = substrings[index];
            if (!counts.ContainsKey(current.Lang)) throw new KeyNotFoundException(current.Lang);
            counts[current.Lang] += current.Length;
            if (index == 0)
            {
                if (substrings[index + 1].Lang is "zh" or "ja" && current.Lang is "zh" or "ja" or "x" && current.Length * 10 < substrings[index + 1].Length)
                {
                    var right = substrings[index + 1];
                    result.Add(new SubString { Lang = right.Lang, Text = current.Text + right.Text, Length = current.Length + right.Length, Index = current.Index });
                    index++;
                }
                else result.Add(current);
            }
            else if (index == substrings.Count - 1)
            {
                var left = result[result.Count - 1];
                if (left.Lang is "zh" or "ja" && current.Lang is "zh" or "ja" or "x" && current.Length * 10 < left.Length)
                {
                    left.Text += current.Text;
                    left.Length += current.Length;
                    index++;
                }
                else result.Add(current);
            }
            else
            {
                var left = result[result.Count - 1];
                var right = substrings[index + 1];
                if (left.Lang == right.Lang && left.Lang is "zh" or "ja" && current.Lang != "en" && current.Length * 10 < left.Length + right.Length)
                {
                    left.Text += current.Text + right.Text;
                    left.Length += current.Length + right.Length;
                    index++;
                }
                else result.Add(current);
            }
            index++;
        }
        if (counts["x"] > 0)
        {
            // max() over the dict in insertion order: the first language with the largest length.
            string max = counts.Aggregate((a, b) => b.Value > a.Value ? b : a).Key;
            foreach (var s in result) if (s.Lang == "x") s.Lang = max;
        }
        if (counts["ja"] >= counts["zh"] * 10)
            foreach (var s in result) if (s.Lang == "zh") s.Lang = "ja";
        return MergeSubstrings(result);
    }

    // The three _merge_substrings_across_*_based_on_sections, which differ only in how the joined piece's language is
    // chosen.
    private List<Section> MergeAcross(List<Section> sections, SectionType kind)
    {
        var merged = new List<Section> { sections[0] };
        for (int index = 1; index < sections.Count; index++)
        {
            var current = sections[index];
            var previous = merged[merged.Count - 1];
            if (kind == SectionType.Digit)
            {
                if (previous.Type == SectionType.Digit || current.Type == SectionType.Digit)
                {
                    if (previous.Type == SectionType.Digit)
                    {
                        if (current.Type != SectionType.Punctuation) previous.Type = current.Type;
                        previous.Text += current.Text;
                        var tail = previous.Substrings[previous.Substrings.Count - 1];
                        tail.Text += current.Substrings[0].Text;
                        tail.Length += current.Substrings[0].Length;
                        tail.Lang = current.Substrings[0].Lang != "punctuation" ? current.Substrings[0].Lang : tail.Lang;
                        previous.Substrings.AddRange(current.Substrings.Skip(1));
                        Reindex(previous.Substrings);
                    }
                    else
                    {
                        previous.Text += current.Text;
                        var tail = previous.Substrings[previous.Substrings.Count - 1];
                        tail.Text += current.Substrings[0].Text;
                        tail.Length += current.Substrings[0].Length;
                        if (tail.Lang == "punctuation") tail.Lang = "digit";
                        if (previous.Type == SectionType.Punctuation) previous.Type = SectionType.Digit;
                    }
                }
                else merged.Add(current);
            }
            else if (kind == SectionType.Punctuation)
            {
                if (previous.Type == SectionType.Punctuation || current.Type == SectionType.Punctuation)
                {
                    if (previous.Type == SectionType.Punctuation && !_notMergePunctuation.Contains(previous.Substrings[0].Text))
                    {
                        previous.Text += current.Text;
                        previous.Type = current.Type;
                        var tail = previous.Substrings[previous.Substrings.Count - 1];
                        tail.Text += current.Substrings[0].Text;
                        tail.Length += current.Substrings[0].Length;
                        tail.Lang = current.Substrings[0].Lang;
                        previous.Substrings.AddRange(current.Substrings.Skip(1));
                        Reindex(previous.Substrings);
                    }
                    else if (current.Type == SectionType.Punctuation && !_notMergePunctuation.Contains(current.Substrings[0].Text))
                    {
                        previous.Text += current.Text;
                        var tail = previous.Substrings[previous.Substrings.Count - 1];
                        tail.Text += current.Substrings[0].Text;
                        tail.Length += current.Substrings[0].Length;
                        previous.Substrings.AddRange(current.Substrings.Skip(1));
                        Reindex(previous.Substrings);
                    }
                    else merged.Add(current);
                }
                else merged.Add(current);
            }
            else
            {
                if (previous.Type == SectionType.Newline)
                {
                    previous.Type = current.Type;
                    previous.Text += current.Text;
                    previous.Substrings.AddRange(current.Substrings);
                    Reindex(previous.Substrings);
                }
                else if (current.Type == SectionType.Newline)
                {
                    previous.Text += current.Text;
                    previous.Substrings.AddRange(current.Substrings);
                    var p = previous.Substrings;
                    p[p.Count - 1].Index = p[p.Count - 2].Index + p[p.Count - 2].Length;
                }
                else merged.Add(current);
            }
        }
        var joined = new List<Section> { merged[0] };
        foreach (var section in merged.Skip(1))
        {
            if (joined[joined.Count - 1].Type == section.Type)
            {
                joined[joined.Count - 1].Text += section.Text;
                joined[joined.Count - 1].Substrings.AddRange(section.Substrings);
            }
            else joined.Add(section);
        }
        for (int s = 0; s < joined.Count; s++)
        {
            var subs = joined[s].Substrings;
            for (int i = 0; i < subs.Count; i++)
            {
                if (i == 0)
                {
                    if (s > 0)
                    {
                        var before = joined[s - 1].Substrings[joined[s - 1].Substrings.Count - 1];
                        subs[0].Index = before.Index + before.Length;
                    }
                }
                else subs[i].Index = subs[i - 1].Index + subs[i - 1].Length;
            }
        }
        foreach (var section in joined) section.Substrings = MergeSubstringsAcross(section.Substrings, kind);
        return joined;
    }

    private static void Reindex(List<SubString> substrings)
    {
        for (int i = 1; i < substrings.Count; i++) substrings[i].Index = substrings[i - 1].Index + substrings[i - 1].Length;
    }

    private static List<SubString> MergeSubstringsAcross(List<SubString> substrings, SectionType kind)
    {
        string marker = kind switch { SectionType.Digit => "digit", SectionType.Punctuation => "punctuation", _ => "newline" };
        var result = new List<SubString>();
        string last = "";
        foreach (var substring in substrings)
        {
            if (result.Count > 0)
            {
                var tail = result[result.Count - 1];
                if (substring.Lang == marker)
                {
                    tail.Text += substring.Text;
                    tail.Length += substring.Length;
                }
                else if (substring.Lang == last || last == "")
                {
                    tail.Text += substring.Text;
                    tail.Length += substring.Length;
                    if (kind == SectionType.Digit)
                    {
                        if (substring.Lang != "punctuation") tail.Lang = substring.Lang;
                    }
                    else if (tail.Lang == marker) tail.Lang = substring.Lang;
                }
                else result.Add(substring);
            }
            else result.Add(substring);
            last = substring.Lang;
        }
        return result;
    }
}

/// <summary>
/// GPT-SoVITS's <c>LangSegmenter.getTexts(text, default_lang)</c> (RVC-Boss/GPT-SoVITS at b4570b8, MIT): split-lang
/// with GPT-SoVITS's language map (digits not merged), then its own rules: English by character class, kana and Hangul
/// runs split out of other text, CJK kept from unknown text, digits and unknown pieces given a neighbour's language.
/// </summary>
internal sealed class GptSoVitsLangSegmenter
{
    private static readonly Dictionary<string, string> LangMap = new(StringComparer.Ordinal)
    {
        ["zh"] = "zh", ["yue"] = "zh", ["wuu"] = "zh", ["zh-cn"] = "zh", ["zh-tw"] = "x", ["ko"] = "ko", ["ja"] = "ja", ["en"] = "en",
    };

    private readonly SplitLang _splitter;

    /// <summary>Uses fastText's lid.176.bin at <paramref name="languageModelPath"/> (fast_langdetect's full model).</summary>
    public GptSoVitsLangSegmenter(string languageModelPath)
    {
        _splitter = new SplitLang(new FastLangDetect(languageModelPath), LangMap, mergeAcrossDigit: false);
    }

    private static readonly Regex FullEnglish = new(@"^(?=.*[A-Za-z])[A-Za-z0-9\s -~\u2000-\u206f\u3000-\u303f\uff00-\uffef]+$");
    private static readonly Regex CjkSymbol = new(@"^[0-9\u3001-\u301c\u3002\uff01\uff1f.!?\u2026 /]+$");
    private static readonly (int From, int To)[] CjkRanges =
    {
        (0x4E00, 0x9FFF), (0x3400, 0x4DB5), (0x20000, 0x2A6DD), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF), (0x30000, 0x3134A), (0x31350, 0x323AF), (0x2EBF0, 0x2EE5D),
    };

    private static bool IsFullEnglish(string text) => FullEnglish.IsMatch(text);

    private static string FullCjk(string text)
    {
        var result = new System.Text.StringBuilder();
        foreach (var c in ChineseText.Codepoints(text))
        {
            int code = char.ConvertToUtf32(c, 0);
            if (CjkRanges.Any(r => code >= r.From && code <= r.To) || CjkSymbol.IsMatch(c)) result.Append(c);
        }
        return result.ToString();
    }

    private static readonly Regex JapaneseRun = new(
        "([\u3041-\u3096\u3099\u309a\u30a1-\u30fa\u30fc]+(?:[0-9\u3001-\u301c\u3002\uff01\uff1f.!?\u2026 ]+[\u3041-\u3096\u3099\u309a\u30a1-\u30fa\u30fc]*)*)");
    private static readonly Regex KoreanRun = new(
        "([\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]+(?:[0-9\u3001-\u301c\u3002\uff01\uff1f.!?\u2026 ]+[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]*)*)");

    private static List<(string Lang, string Text)> SplitJaKo(string tagLang, (string Lang, string Text) item)
    {
        var pattern = tagLang == "ja" ? JapaneseRun : KoreanRun;
        var list = new List<(string, string)>();
        int tag = 0;
        foreach (Match match in pattern.Matches(item.Text))
        {
            if (match.Index > tag) list.Add((item.Lang, item.Text.Substring(tag, match.Index - tag)));
            tag = match.Index + match.Length;
            list.Add((tagLang, match.Value));
        }
        if (tag < item.Text.Length) list.Add((item.Lang, item.Text.Substring(tag)));
        return list;
    }

    private static void Merge(List<(string Lang, string Text)> list, (string Lang, string Text) item)
    {
        if (list.Count > 0 && item.Lang == list[list.Count - 1].Lang) list[list.Count - 1] = (item.Lang, list[list.Count - 1].Text + item.Text);
        else list.Add(item);
    }

    private static readonly string[] Stops = { ",", ".", "!", "?", "\uff0c", "\u3002", "\uff01", "\uff1f" };

    private static string LastChar(string s) => ChineseText.Codepoints(s).Last();

    private static string FirstChar(string s) => ChineseText.Codepoints(s).First();

    /// <summary><c>LangSegmenter.getTexts(text, default_lang)</c>: (language, text) pieces.</summary>
    public List<(string Lang, string Text)> GetTexts(string text, string defaultLang = "")
    {
        var list = new List<(string Lang, string Text)>();
        bool haveNumber = false;
        foreach (var piece in _splitter.SplitByLang(text))
        {
            var item = (Lang: piece.Lang, Text: piece.Text);
            if (item.Lang == "digit")
            {
                if (defaultLang != "") item.Lang = defaultLang;
                else haveNumber = true;
                Merge(list, item);
                continue;
            }
            if (IsFullEnglish(item.Text))
            {
                Merge(list, ("en", item.Text));
                continue;
            }
            if (defaultLang != "")
            {
                Merge(list, (defaultLang, item.Text));
                continue;
            }
            var jaList = item.Lang != "ja" ? SplitJaKo("ja", item) : new List<(string, string)>();
            if (jaList.Count == 0) jaList.Add(item);
            var koList = new List<(string Lang, string Text)>();
            var temp = new List<(string Lang, string Text)>();
            foreach (var koItem in jaList)
            {
                if (koItem.Item1 != "ko") koList = SplitJaKo("ko", koItem);
                if (koList.Count > 0) temp.AddRange(koList);
                else temp.Add(koItem);
            }
            if (temp.Count == 1)
            {
                if (item.Lang == "x")
                {
                    string cjk = FullCjk(item.Text);
                    Merge(list, cjk.Length > 0 ? ("zh", cjk) : item);
                }
                else Merge(list, item);
                continue;
            }
            foreach (var t in temp)
            {
                if (t.Lang == "x")
                {
                    string cjk = FullCjk(t.Text);
                    Merge(list, cjk.Length > 0 ? ("zh", cjk) : t);
                }
                else Merge(list, t);
            }
        }
        if (haveNumber)
        {
            var temp = list;
            list = new List<(string Lang, string Text)>();
            for (int i = 0; i < temp.Count; i++)
            {
                var t = temp[i];
                if (t.Lang == "digit")
                {
                    if (defaultLang != "") t.Lang = defaultLang;
                    else if (list.Count > 0 && i == temp.Count - 1) t.Lang = list[list.Count - 1].Lang;
                    else if (list.Count == 0 && i < temp.Count - 1) t.Lang = temp[1].Lang;
                    else if (list.Count > 0 && i < temp.Count - 1)
                    {
                        var previous = list[list.Count - 1];
                        var next = temp[i + 1];
                        if (previous.Lang == next.Lang) t.Lang = previous.Lang;
                        else if (Stops.Contains(LastChar(previous.Text))) t.Lang = next.Lang;
                        else if (Stops.Contains(FirstChar(next.Text))) t.Lang = previous.Lang;
                        else if (LastChar(t.Text) is "\u3002" or ".") t.Lang = previous.Lang;
                        else if (ChineseText.Codepoints(previous.Text).Count >= ChineseText.Codepoints(next.Text).Count) t.Lang = previous.Lang;
                        else t.Lang = next.Lang;
                    }
                    else t.Lang = "zh";
                }
                Merge(list, t);
            }
        }
        var filtered = new List<(string Lang, string Text)>();
        for (int i = 0; i < list.Count; i++)
        {
            var t = list[i];
            if (t.Lang == "x")
            {
                if (filtered.Count > 0) t.Lang = filtered[filtered.Count - 1].Lang;
                else if (list.Count > 1) t.Lang = list[1].Lang;
                else t.Lang = "zh";
            }
            Merge(filtered, t);
        }
        return filtered;
    }
}
