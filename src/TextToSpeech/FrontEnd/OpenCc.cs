using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// OpenCC 1.4.2's <c>s2tw</c> configuration (Simplified Chinese to Taiwan Traditional; Apache-2.0), as its Python binding's
/// <c>OpenCC("s2tw").convert</c> runs it: the CJK compatibility ideographs normalized, the text cut by maximum matching on
/// the phrase dictionaries, and each piece converted by the longest dictionary match at each position, first by the
/// phrases or else the characters, then by the Taiwan variants. G2PW reads Traditional characters, so GPT-SoVITS converts
/// its Simplified input with this before predicting.
/// </summary>
/// <remarks>
/// The dictionaries are the installed package's own, dumped by <c>tools/reference-data/gptsovits_chinese_resources.py</c>.
/// Text is matched by code point (OpenCC matches UTF-8 bytes; whole characters match identically). An ideographic
/// description sequence (U+2FF0–U+2FFF and its operands) is kept whole and unconverted, as OpenCC keeps it.
/// </remarks>
internal sealed class OpenCc
{
    private static readonly Lazy<OpenCc> S2TwInstance = new(() => new OpenCc(), isThreadSafe: true);

    /// <summary>The <c>s2tw</c> converter (the dictionaries load once, on first use).</summary>
    public static OpenCc S2Tw => S2TwInstance.Value;

    private interface IMatcher
    {
        // The longest key that prefixes text[start..end): its length in code points and its first value.
        (int Length, string Value)? MatchPrefix(List<string> text, int start, int end);
    }

    private sealed class TextDictionary : IMatcher
    {
        private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);
        private readonly HashSet<string> _firstCharacters = new(StringComparer.Ordinal);
        private int _maxLength;

        public void Add(string key, string value)
        {
            var codepoints = ChineseText.Codepoints(key);
            if (codepoints.Count == 0) return;
            _values[key] = value;
            _firstCharacters.Add(codepoints[0]);
            _maxLength = Math.Max(_maxLength, codepoints.Count);
        }

        public (int Length, string Value)? MatchPrefix(List<string> text, int start, int end)
        {
            if (start >= end || !_firstCharacters.Contains(text[start])) return null;
            var key = new StringBuilder();
            (int, string)? best = null;
            for (int length = 1; length <= _maxLength && start + length <= end; length++)
            {
                key.Append(text[start + length - 1]);
                if (_values.TryGetValue(key.ToString(), out var value)) best = (length, value);
            }
            return best;
        }
    }

    // match_policy "union": the longest match over the dictionaries (the first on a tie).
    private sealed class UnionGroup : IMatcher
    {
        private readonly IMatcher[] _dictionaries;

        public UnionGroup(params IMatcher[] dictionaries) => _dictionaries = dictionaries;

        public (int Length, string Value)? MatchPrefix(List<string> text, int start, int end)
        {
            (int Length, string Value)? best = null;
            foreach (var dictionary in _dictionaries)
            {
                var match = dictionary.MatchPrefix(text, start, end);
                if (match is { } m && (best is null || m.Length > best.Value.Length)) best = m;
            }
            return best;
        }
    }

    // match_policy "short_circuit": the first dictionary that matches at all.
    private sealed class ShortCircuitGroup : IMatcher
    {
        private readonly IMatcher[] _dictionaries;

        public ShortCircuitGroup(params IMatcher[] dictionaries) => _dictionaries = dictionaries;

        public (int Length, string Value)? MatchPrefix(List<string> text, int start, int end)
        {
            foreach (var dictionary in _dictionaries)
                if (dictionary.MatchPrefix(text, start, end) is { } match) return match;
            return null;
        }
    }

    private readonly IMatcher _normalization;
    private readonly IMatcher _segmentation;
    private readonly IMatcher[] _chain;

    private OpenCc()
    {
        var dictionaries = new Dictionary<string, TextDictionary>(StringComparer.Ordinal);
        TextDictionary? current = null;
        foreach (var line in ChineseText.ReadLines("opencc_s2tw.tsv.gz"))
        {
            if (line[0] == '#')
            {
                current = new TextDictionary();
                dictionaries[line.Substring(1)] = current;
                continue;
            }
            int tab = line.IndexOf('\t');
            string values = line.Substring(tab + 1);
            int space = values.IndexOf(' ');
            current!.Add(line.Substring(0, tab), space < 0 ? values : values.Substring(0, space));
        }
        // s2tw.json: normalization, segmentation and the conversion chain.
        _normalization = dictionaries["CJK_Compatibility_Ideographs"];
        _segmentation = new UnionGroup(dictionaries["STPhrases"], dictionaries["STPhrases_GeneratedFromRegionalPhrases"]);
        _chain = new IMatcher[]
        {
            new ShortCircuitGroup(new UnionGroup(dictionaries["STPhrases"], dictionaries["STPhrases_GeneratedFromRegionalPhrases"]),
                dictionaries["STCharacters"]),
            new ShortCircuitGroup(dictionaries["TWVariantsPhrases"], dictionaries["TWVariants"]),
        };
    }

    /// <summary><c>OpenCC("s2tw").convert(text)</c>.</summary>
    public string Convert(string text)
    {
        var normalized = ChineseText.Codepoints(Apply(_normalization, ChineseText.Codepoints(text)));
        var output = new StringBuilder();
        foreach (var segment in Segment(normalized))
        {
            string converted = string.Concat(segment);
            foreach (var conversion in _chain) converted = Apply(conversion, ChineseText.Codepoints(converted));
            output.Append(converted);
        }
        return output.ToString();
    }

    // MaxMatchSegmentation: each dictionary match is a segment; unmatched characters (an ideographic description sequence
    // counting as one) run together into one.
    private IEnumerable<List<string>> Segment(List<string> text)
    {
        var pending = new List<string>();
        int at = 0;
        while (at < text.Count)
        {
            if (_segmentation.MatchPrefix(text, at, text.Count) is { } match)
            {
                if (pending.Count > 0)
                {
                    yield return pending;
                    pending = new List<string>();
                }
                yield return text.GetRange(at, match.Length);
                at += match.Length;
                continue;
            }
            int length = Math.Max(1, IdeographicDescriptionSequenceLength(text, at, text.Count));
            pending.AddRange(text.GetRange(at, length));
            at += length;
        }
        if (pending.Count > 0) yield return pending;
    }

    // Conversion::AppendConverted: at each position the longest match's value, else the character (or ideographic
    // description sequence) as written.
    private static string Apply(IMatcher dictionary, List<string> text)
    {
        var output = new StringBuilder();
        int at = 0;
        while (at < text.Count)
        {
            if (dictionary.MatchPrefix(text, at, text.Count) is { } match)
            {
                output.Append(match.Value);
                at += match.Length;
                continue;
            }
            int length = Math.Max(1, IdeographicDescriptionSequenceLength(text, at, text.Count));
            for (int i = 0; i < length; i++) output.Append(text[at + i]);
            at += length;
        }
        return output.ToString();
    }

    private enum IdsStatus { Complete, Incomplete, Invalid }

    private static int OperatorArity(string codepoint)
    {
        int c = char.ConvertToUtf32(codepoint, 0);
        return c switch
        {
            0x2FF2 or 0x2FF3 => 3,
            0x2FFE or 0x2FFF => 1,
            >= 0x2FF0 and <= 0x2FFD => 2,
            _ => 0,
        };
    }

    // UTF8Util::NextIdeographicDescriptionSequenceLength: the code points of a complete sequence starting with an
    // operator (at most 16 deep and 64 code points), else 0.
    private static int IdeographicDescriptionSequenceLength(List<string> text, int start, int end)
    {
        if (start >= end || OperatorArity(text[start]) == 0) return 0;
        int codepoints = 0;
        return Consume(text, start, end, 16, ref codepoints, out int consumed) == IdsStatus.Complete ? consumed : 0;
    }

    private static IdsStatus Consume(List<string> text, int start, int end, int depthLeft, ref int codepoints, out int consumed)
    {
        consumed = 0;
        if (start >= end) return IdsStatus.Incomplete;
        if (depthLeft == 0 || codepoints >= 64) return IdsStatus.Invalid;
        codepoints++;
        int arity = OperatorArity(text[start]);
        if (arity == 0)
        {
            consumed = 1;
            return IdsStatus.Complete;
        }
        int offset = 1;
        for (int i = 0; i < arity; i++)
        {
            if (start + offset >= end) return IdsStatus.Incomplete;
            var status = Consume(text, start + offset, end, depthLeft - 1, ref codepoints, out int operand);
            if (status != IdsStatus.Complete) return status;
            offset += operand;
        }
        consumed = offset;
        return IdsStatus.Complete;
    }
}
