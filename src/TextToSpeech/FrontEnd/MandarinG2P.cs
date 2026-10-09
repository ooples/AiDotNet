using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Converts Mandarin Chinese text to IPA phonemes, as VALL-E X's reference front end does (Plachtaa/VALL-E-X,
/// <c>utils/g2p/mandarin.py</c>, <c>chinese_to_ipa</c>).
/// </summary>
/// <remarks>
/// <para>
/// The pipeline is the reference's: Arabic numerals become Chinese numerals (cn2an's <c>an2cn</c>, "low" mode), jieba
/// segments the text into words (its prefix dictionary, maximum-probability route and HMM for unknown words), pypinyin
/// reads each word in BOPOMOFO (its phrase dictionary first, by forward maximum matching, then each character's first
/// reading), Latin letters are spelled in bopomofo, and a table maps bopomofo to IPA with tone contours
/// (→ ↑ ↓↑ ↓), followed by the reference's four rewrite rules (glides, apical vowels). The dictionaries are jieba 0.42.1's
/// and pypinyin 0.55.0's (MIT), exported by <c>tools/reference-data/mandarin_g2p_resources.py</c>.
/// </para>
/// <para>
/// <see cref="Phonemize"/> splits the IPA into the reference tokenizer's base symbols: one symbol per character, with
/// spaces as the word separator <c>_</c>.
/// </para>
/// <para><b>For Beginners:</b> Chinese characters do not spell their sounds, so this looks words up in a dictionary to
/// find each character's pronunciation (and tone) and writes it in the International Phonetic Alphabet.</para>
/// </remarks>
public sealed class MandarinG2P
{
    private static readonly Lazy<MandarinG2P> SharedInstance = new(() => new MandarinG2P(), isThreadSafe: true);

    // jieba
    private readonly Dictionary<string, long> _frequency = new(StringComparer.Ordinal);
    private readonly double _logTotal;
    private readonly Dictionary<char, double> _hmmStart = new();
    private readonly Dictionary<char, Dictionary<char, double>> _hmmTransition = new();
    private readonly Dictionary<char, Dictionary<string, double>> _hmmEmission = new();

    // pypinyin
    private readonly Dictionary<string, string[]> _phrases = new(StringComparer.Ordinal);
    private readonly HashSet<string> _phrasePrefixes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _characters = new(StringComparer.Ordinal);

    private MandarinG2P()
    {
        long total = 0;
        foreach (var line in ReadLines("jieba_dict.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            string word = line.Substring(0, tab);
            long frequency = long.Parse(line.Substring(tab + 1), CultureInfo.InvariantCulture);
            _frequency[word] = frequency;
            total += frequency;
            var codepoints = Codepoints(word);
            var prefix = new StringBuilder();
            for (int i = 0; i < codepoints.Count - 1; i++)
            {
                prefix.Append(codepoints[i]);
                string fragment = prefix.ToString();
                if (!_frequency.ContainsKey(fragment)) _frequency[fragment] = 0;
            }
        }
        _logTotal = Math.Log(total);

        var hmm = JObject.Parse(string.Join("\n", ReadLines("jieba_hmm.json.gz")));
        foreach (var state in (JObject)hmm["start"]!)
            _hmmStart[state.Key[0]] = (double)state.Value!;
        foreach (var from in (JObject)hmm["trans"]!)
        {
            var row = new Dictionary<char, double>();
            foreach (var to in (JObject)from.Value!) row[to.Key[0]] = (double)to.Value!;
            _hmmTransition[from.Key[0]] = row;
        }
        foreach (var state in (JObject)hmm["emit"]!)
        {
            var row = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var emission in (JObject)state.Value!) row[emission.Key] = (double)emission.Value!;
            _hmmEmission[state.Key[0]] = row;
        }

        foreach (var line in ReadLines("pypinyin_chars.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            _characters[line.Substring(0, tab)] = line.Substring(tab + 1);
        }
        foreach (var line in ReadLines("pypinyin_phrases.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            string phrase = line.Substring(0, tab);
            _phrases[phrase] = line.Substring(tab + 1).Split(' ');
            var codepoints = Codepoints(phrase);
            var prefix = new StringBuilder();
            foreach (var c in codepoints)
            {
                prefix.Append(c);
                _phrasePrefixes.Add(prefix.ToString());
            }
        }
    }

    /// <summary>Every symbol <see cref="Phonemize"/> produces from Chinese, punctuation and Latin letters (characters it
    /// passes through unchanged, such as other symbols, are outside it).</summary>
    public static IReadOnlyList<string> Symbols => SymbolTable.Value;

    /// <summary>The shared instance (the dictionaries load once, on first use).</summary>
    public static MandarinG2P Default => SharedInstance.Value;

    private static IEnumerable<string> ReadLines(string resource)
    {
        var assembly = typeof(MandarinG2P).Assembly;
        string name = assembly.GetManifestResourceNames().Single(n => n.EndsWith(resource, StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(name)!;
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var lines = new List<string>();
        for (string? line = reader.ReadLine(); line is not null; line = reader.ReadLine())
            if (line.Length > 0) lines.Add(line);
        return lines;
    }

    // Python iterates strings by code point; supplementary characters are surrogate pairs here.
    private static List<string> Codepoints(string text)
    {
        var result = new List<string>(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                result.Add(text.Substring(i, 2));
                i++;
            }
            else
            {
                result.Add(text[i].ToString());
            }
        }
        return result;
    }

    /// <summary>The phonemes of <paramref name="text"/>: one symbol per IPA character, words separated by <c>_</c>.</summary>
    public IReadOnlyList<string> Phonemize(string text) =>
        Codepoints(ToIpa(text)).Select(c => c == " " ? "_" : c).ToList();

    /// <summary>The reference's <c>chinese_to_ipa</c> of <paramref name="text"/>.</summary>
    public string ToIpa(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        text = NumbersToChinese(text);
        text = ToBopomofo(text);
        text = LatinToBopomofo(text);
        foreach (var (pattern, replacement) in BopomofoToIpa) text = text.Replace(pattern, replacement);
        text = Regex.Replace(text, "i([aoe])", "j$1");
        text = Regex.Replace(text, "u([aoəe])", "w$1");
        text = Regex.Replace(text, "([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", "$1ɹ`$2").Replace("ɻ", "ɹ`");
        text = Regex.Replace(text, "([s][⁼ʰ]?)([→↓↑ ]+|$)", "$1ɹ$2");
        return text;
    }

    // ---------------------------------------------------------------- numerals (cn2an)

    private static readonly string[] Digits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
    private static readonly string[] Units = { "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千" };

    // number_to_chinese: each \d+(?:\.?\d+)? in turn, replacing its first occurrence.
    private static string NumbersToChinese(string text)
    {
        foreach (Match match in Regex.Matches(text, @"\d+(?:\.?\d+)?"))
        {
            int index = text.IndexOf(match.Value, StringComparison.Ordinal);
            text = text.Substring(0, index) + AnToCn(match.Value) + text.Substring(index + match.Value.Length);
        }
        return text;
    }

    /// <summary>cn2an's <c>an2cn</c> in its default "low" mode.</summary>
    internal static string AnToCn(string number)
    {
        foreach (char c in number)
            if (!(c is >= '0' and <= '9' || c == '.' || c == '-'))
                throw new ArgumentException($"'{number}' is outside cn2an's range (it reads ASCII digits).", nameof(number));
        string sign = number.StartsWith("-", StringComparison.Ordinal) ? "负" : "";
        if (sign.Length > 0) number = number.Substring(1);
        var parts = number.Split('.');
        if (parts.Length > 2) throw new ArgumentException($"Malformed number '{number}'.", nameof(number));
        string output = IntegerToCn(parts[0]);
        if (parts.Length == 2)
        {
            string decimals = parts[1].Length > 16 ? parts[1].Substring(0, 16) : parts[1];
            output += decimals.Length > 0 ? "点" : "";
            foreach (char d in decimals) output += Digits[d - '0'];
        }
        return sign + output;
    }

    private static string IntegerToCn(string digits)
    {
        // str(int(...)): leading zeros dropped (an empty integer part is int("") in Python, an error).
        if (digits.Length == 0) throw new ArgumentException("cn2an needs an integer part.", nameof(digits));
        digits = digits.TrimStart('0');
        if (digits.Length == 0) digits = "0";
        int length = digits.Length;
        if (length > Units.Length) throw new ArgumentException($"cn2an reads at most {Units.Length} integer digits.", nameof(digits));
        var output = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            int d = digits[i] - '0';
            if (d != 0)
            {
                output.Append(Digits[d]).Append(Units[length - i - 1]);
            }
            else
            {
                if ((length - i - 1) % 4 == 0) output.Append(Digits[d]).Append(Units[length - i - 1]);
                if (i > 0 && (output.Length == 0 || output[output.Length - 1] != '零')) output.Append(Digits[d]);
            }
        }
        string result = output.ToString().Replace("零零", "零").Replace("零万", "万").Replace("零亿", "亿").Replace("亿万", "亿").Trim('零');
        result = Regex.Replace(result, "([万亿])零([一二三四五六七八九壹贰叁肆伍陆柒捌玖][千仟])", "$1$2");
        if (result.StartsWith("一十", StringComparison.Ordinal)) result = result.Substring(1);
        return result.Length == 0 ? "零" : result;
    }

    // ---------------------------------------------------------------- words and readings

    // chinese_to_bopomofo: jieba's words, each read by pypinyin; words without a CJK character pass through.
    private string ToBopomofo(string text)
    {
        text = text.Replace('、', '，').Replace('；', '，').Replace('：', '，');
        var output = new StringBuilder();
        foreach (var word in JiebaCut(text))
        {
            if (!Regex.IsMatch(word, "[一-鿿]"))
            {
                output.Append(word);
                continue;
            }
            if (output.Length > 0) output.Append(' ');
            foreach (var reading in LazyPinyin(word))
                output.Append(Regex.Replace(reading, "([ㄅ-ㄩ])$", "$1ˉ"));
        }
        return output.ToString();
    }

    private static readonly (string Letter, string Bopomofo)[] LatinBopomofo =
    {
        ("a", "ㄟˉ"), ("b", "ㄅㄧˋ"), ("c", "ㄙㄧˉ"), ("d", "ㄉㄧˋ"), ("e", "ㄧˋ"), ("f", "ㄝˊㄈㄨˋ"), ("g", "ㄐㄧˋ"),
        ("h", "ㄝˇㄑㄩˋ"), ("i", "ㄞˋ"), ("j", "ㄐㄟˋ"), ("k", "ㄎㄟˋ"), ("l", "ㄝˊㄛˋ"), ("m", "ㄝˊㄇㄨˋ"), ("n", "ㄣˉ"),
        ("o", "ㄡˉ"), ("p", "ㄆㄧˉ"), ("q", "ㄎㄧㄡˉ"), ("r", "ㄚˋ"), ("s", "ㄝˊㄙˋ"), ("t", "ㄊㄧˋ"), ("u", "ㄧㄡˉ"),
        ("v", "ㄨㄧˉ"), ("w", "ㄉㄚˋㄅㄨˋㄌㄧㄡˋ"), ("x", "ㄝˉㄎㄨˋㄙˋ"), ("y", "ㄨㄞˋ"), ("z", "ㄗㄟˋ"),
    };

    private static string LatinToBopomofo(string text)
    {
        foreach (var (letter, bopomofo) in LatinBopomofo)
            text = Regex.Replace(text, letter, bopomofo, RegexOptions.IgnoreCase);
        return text;
    }

    private static readonly (string Pattern, string Replacement)[] BopomofoToIpa =
    {
        ("ㄅㄛ", "p⁼wo"), ("ㄆㄛ", "pʰwo"), ("ㄇㄛ", "mwo"), ("ㄈㄛ", "fwo"), ("ㄅ", "p⁼"), ("ㄆ", "pʰ"), ("ㄇ", "m"),
        ("ㄈ", "f"), ("ㄉ", "t⁼"), ("ㄊ", "tʰ"), ("ㄋ", "n"), ("ㄌ", "l"), ("ㄍ", "k⁼"), ("ㄎ", "kʰ"), ("ㄏ", "x"),
        ("ㄐ", "tʃ⁼"), ("ㄑ", "tʃʰ"), ("ㄒ", "ʃ"), ("ㄓ", "ts`⁼"), ("ㄔ", "ts`ʰ"), ("ㄕ", "s`"), ("ㄖ", "ɹ`"),
        ("ㄗ", "ts⁼"), ("ㄘ", "tsʰ"), ("ㄙ", "s"), ("ㄚ", "a"), ("ㄛ", "o"), ("ㄜ", "ə"), ("ㄝ", "ɛ"), ("ㄞ", "aɪ"),
        ("ㄟ", "eɪ"), ("ㄠ", "ɑʊ"), ("ㄡ", "oʊ"), ("ㄧㄢ", "jɛn"), ("ㄩㄢ", "ɥæn"), ("ㄢ", "an"), ("ㄧㄣ", "in"),
        ("ㄩㄣ", "ɥn"), ("ㄣ", "ən"), ("ㄤ", "ɑŋ"), ("ㄧㄥ", "iŋ"), ("ㄨㄥ", "ʊŋ"), ("ㄩㄥ", "jʊŋ"), ("ㄥ", "əŋ"),
        ("ㄦ", "əɻ"), ("ㄧ", "i"), ("ㄨ", "u"), ("ㄩ", "ɥ"), ("ˉ", "→"), ("ˊ", "↑"), ("ˇ", "↓↑"), ("ˋ", "↓"),
        ("˙", ""), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("—", "-"),
    };

    // Declared after the table it reads: static fields initialize in textual order.
    private static readonly Lazy<IReadOnlyList<string>> SymbolTable = new(() =>
    {
        var symbols = new SortedSet<string>(StringComparer.Ordinal) { "_", "j", "w", "ɹ", "`" };
        foreach (var (_, replacement) in BopomofoToIpa)
            foreach (var c in Codepoints(replacement)) symbols.Add(c);
        return symbols.ToList();
    }, isThreadSafe: true);

    // pypinyin's RE_HANS character ranges.
    private static readonly (int From, int To)[] HanRanges =
    {
        (0x3007, 0x3007), (0xE815, 0xE864), (0xFA18, 0xFA18), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF), (0x2A703, 0x2B73F), (0x2B740, 0x2B81D), (0x2B825, 0x2BF6E), (0x2C029, 0x2CE93),
        (0x2D016, 0x2D016), (0x2D11B, 0x2EBD9), (0x2F80A, 0x2FA1F), (0x30000, 0x3134A), (0x300F7, 0x31288),
        (0x30EDD, 0x30EDE), (0x31350, 0x32389),
    };

    private static bool IsHan(string codepoint)
    {
        int value = char.ConvertToUtf32(codepoint, 0);
        foreach (var (from, to) in HanRanges)
            if (value >= from && value <= to) return true;
        return false;
    }

    // lazy_pinyin(word, BOPOMOFO): simple_seg into Han and other runs, the Han runs cut by forward maximum matching on
    // the phrase dictionary (no_non_phrases), each piece read as a phrase or character by character; a run without a
    // reading is kept as written.
    private IEnumerable<string> LazyPinyin(string word)
    {
        var codepoints = Codepoints(word);
        var runs = new List<(bool Han, List<string> Text)>();
        foreach (var c in codepoints)
        {
            bool han = IsHan(c);
            if (runs.Count == 0 || runs[runs.Count - 1].Han != han) runs.Add((han, new List<string>()));
            runs[runs.Count - 1].Text.Add(c);
        }
        foreach (var (han, text) in runs)
        {
            if (!han)
            {
                yield return string.Concat(text);
                continue;
            }
            foreach (var piece in MaximumMatching(text))
                foreach (var reading in Convert(piece)) yield return reading;
        }
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
            // The whole remainder is a prefix.
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

    // DefaultConverter.convert(word, BOPOMOFO, heteronym=False, errors="default"): the phrase's readings, or each
    // character's (a character without a reading is kept as written). mmseg yields phrases and single characters only.
    private IEnumerable<string> Convert(List<string> piece)
    {
        if (piece.Count > 1 && _phrases.TryGetValue(string.Concat(piece), out var readings))
            return readings;
        return piece.Select(c => _characters.TryGetValue(c, out var reading) ? reading : c);
    }

    // ---------------------------------------------------------------- jieba

    private static readonly Regex JiebaHan = new("([一-鿕a-zA-Z0-9+#&\\._%\\-]+)");
    private static readonly Regex JiebaSkip = new("(\r\n|\\s)");
    private static readonly Regex HmmHan = new("([一-鿕]+)");
    private static readonly Regex HmmSkip = new("([a-zA-Z0-9]+(?:\\.\\d+)?%?)");

    /// <summary>jieba's <c>lcut(text, cut_all=False)</c> (HMM on).</summary>
    internal List<string> JiebaCut(string sentence)
    {
        var words = new List<string>();
        foreach (var block in JiebaHan.Split(sentence))
        {
            if (block.Length == 0) continue;
            if (JiebaHan.IsMatch(block) && JiebaHan.Match(block).Index == 0)
            {
                words.AddRange(CutDag(Codepoints(block)));
                continue;
            }
            foreach (var x in JiebaSkip.Split(block))
            {
                if (x.Length == 0) continue;
                if (JiebaSkip.Match(x) is { Success: true, Index: 0 }) words.Add(x);
                else words.AddRange(Codepoints(x));
            }
        }
        return words;
    }

    private long? Frequency(string word) => _frequency.TryGetValue(word, out long f) ? f : null;

    // Tokenizer.__cut_DAG
    private IEnumerable<string> CutDag(List<string> sentence)
    {
        int n = sentence.Count;
        string Slice(int from, int to) => string.Concat(sentence.GetRange(from, to - from));
        // get_DAG
        var dag = new List<int>[n];
        for (int k = 0; k < n; k++)
        {
            var ends = new List<int>();
            int i = k;
            string fragment = sentence[k];
            while (i < n && _frequency.TryGetValue(fragment, out long f))
            {
                if (f > 0) ends.Add(i);
                i++;
                if (i < n) fragment = Slice(k, i + 1);
            }
            if (ends.Count == 0) ends.Add(k);
            dag[k] = ends;
        }
        // calc: route[idx] = max over x of (log(FREQ[sentence[idx:x+1]] or 1) − log total + route[x+1].0, x)
        var route = new (double Score, int End)[n + 1];
        route[n] = (0, 0);
        for (int idx = n - 1; idx >= 0; idx--)
        {
            (double Score, int End) best = (double.NegativeInfinity, -1);
            foreach (int x in dag[idx])
            {
                long f = Frequency(Slice(idx, x + 1)) ?? 0;
                double score = Math.Log(f == 0 ? 1 : f) - _logTotal + route[x + 1].Score;
                if (score > best.Score || (score == best.Score && x > best.End)) best = (score, x);
            }
            route[idx] = best;
        }
        var buffer = new StringBuilder();
        int bufferCount = 0;
        IEnumerable<string> Flush()
        {
            if (bufferCount == 0) yield break;
            string pending = buffer.ToString();
            if (bufferCount == 1) yield return pending;
            else if (Frequency(pending) is not > 0)
                foreach (var t in FinalSegCut(pending)) yield return t;
            else
                foreach (var c in Codepoints(pending)) yield return c;
            buffer.Clear();
            bufferCount = 0;
        }
        int at = 0;
        while (at < n)
        {
            int y = route[at].End + 1;
            if (y - at == 1)
            {
                buffer.Append(sentence[at]);
                bufferCount++;
            }
            else
            {
                foreach (var t in Flush()) yield return t;
                yield return Slice(at, y);
            }
            at = y;
        }
        foreach (var t in Flush()) yield return t;
    }

    // finalseg.cut
    private IEnumerable<string> FinalSegCut(string sentence)
    {
        foreach (var block in HmmHan.Split(sentence))
        {
            if (block.Length == 0) continue;
            if (HmmHan.Match(block) is { Success: true, Index: 0 } m && m.Length == block.Length)
            {
                foreach (var word in Viterbi(Codepoints(block))) yield return word;
                continue;
            }
            foreach (var x in HmmSkip.Split(block))
                if (x.Length > 0) yield return x;
        }
    }

    private const double MinFloat = -3.14e100;
    private static readonly Dictionary<char, string> PreviousStates = new()
    {
        ['B'] = "ES", ['M'] = "MB", ['S'] = "SE", ['E'] = "BM",
    };

    // finalseg.__cut: the B/M/E/S Viterbi path, then words from it.
    private IEnumerable<string> Viterbi(List<string> observations)
    {
        const string states = "BMES";
        double Emit(char state, string observation) =>
            _hmmEmission[state].TryGetValue(observation, out double p) ? p : MinFloat;
        double Transition(char from, char to) =>
            _hmmTransition[from].TryGetValue(to, out double p) ? p : MinFloat;
        var probability = new Dictionary<char, double>();
        var path = new Dictionary<char, List<char>>();
        foreach (char y in states)
        {
            probability[y] = _hmmStart[y] + Emit(y, observations[0]);
            path[y] = new List<char> { y };
        }
        for (int t = 1; t < observations.Count; t++)
        {
            var next = new Dictionary<char, double>();
            var nextPath = new Dictionary<char, List<char>>();
            foreach (char y in states)
            {
                double emission = Emit(y, observations[t]);
                (double Score, char State) best = (double.NegativeInfinity, '\0');
                foreach (char y0 in PreviousStates[y])
                {
                    double score = probability[y0] + Transition(y0, y) + emission;
                    if (score > best.Score || (score == best.Score && y0 > best.State)) best = (score, y0);
                }
                next[y] = best.Score;
                nextPath[y] = new List<char>(path[best.State]) { y };
            }
            probability = next;
            path = nextPath;
        }
        // max((V[-1][y], y) for y in "ES"): a tie goes to "S".
        char last = probability['E'] > probability['S'] ? 'E' : 'S';
        var positions = path[last];
        int begin = 0, nextIndex = 0;
        for (int i = 0; i < observations.Count; i++)
        {
            char position = positions[i];
            if (position == 'B')
            {
                begin = i;
            }
            else if (position == 'E')
            {
                yield return string.Concat(observations.GetRange(begin, i - begin + 1));
                nextIndex = i + 1;
            }
            else if (position == 'S')
            {
                yield return observations[i];
                nextIndex = i + 1;
            }
        }
        if (nextIndex < observations.Count) yield return string.Concat(observations.GetRange(nextIndex, observations.Count - nextIndex));
    }
}
