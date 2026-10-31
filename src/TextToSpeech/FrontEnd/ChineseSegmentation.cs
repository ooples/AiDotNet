using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Reading the Chinese front ends' embedded resources, and Python's code-point view of a string.</summary>
internal static class ChineseText
{
    internal static IEnumerable<string> ReadLines(string resource)
    {
        var assembly = typeof(ChineseText).Assembly;
        string name = assembly.GetManifestResourceNames().Single(n => n.EndsWith("." + resource, StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(name)!;
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var lines = new List<string>();
        for (string? line = reader.ReadLine(); line is not null; line = reader.ReadLine())
            if (line.Length > 0) lines.Add(line);
        return lines;
    }

    // Python iterates strings by code point; supplementary characters are surrogate pairs here.
    internal static List<string> Codepoints(string text)
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
}

/// <summary>
/// jieba 0.42.1 (MIT): <c>lcut</c> (its prefix dictionary's maximum-probability route, with the B/M/E/S HMM for unknown
/// words), <c>lcut_for_search</c> (plus the dictionary's 2- and 3-character sub-words) and <c>posseg.lcut</c> (the same
/// route, words tagged from the dictionary, unknown runs segmented and tagged by the part-of-speech HMM). The dictionaries
/// are jieba's own, exported by <c>tools/reference-data/mandarin_g2p_resources.py</c> and
/// <c>melo_chinese_resources.py</c>.
/// </summary>
/// <remarks>
/// jieba_fast 0.53 (GPT-SoVITS's segmenter; the same dictionaries) replaces <c>lcut</c>'s route and unknown-word HMM with
/// C, which this class follows when asked (<c>jiebaFast</c>): '-' does not join a Han block, each position keeps at most
/// 12 dictionary words, a tie in the route goes to the shorter word, and the HMM sums its terms in another order, keeps
/// the first predecessor on a tie and ends on 'E' on a tie. Its <c>posseg</c> is jieba's Python, unchanged.
/// </remarks>
internal sealed class Jieba
{
    private static readonly Lazy<Jieba> SharedInstance = new(() => new Jieba(), isThreadSafe: true);

    private readonly Dictionary<string, long> _frequency = new(StringComparer.Ordinal);
    private readonly double _logTotal;
    private readonly Dictionary<char, double> _hmmStart = new();
    private readonly Dictionary<char, Dictionary<char, double>> _hmmTransition = new();
    private readonly Dictionary<char, Dictionary<string, double>> _hmmEmission = new();
    private readonly Lazy<Dictionary<string, string>> _tags;
    private readonly Lazy<PosModel> _posModel;

    /// <summary>The shared instance (the dictionary loads once, on first use).</summary>
    public static Jieba Default => SharedInstance.Value;

    private Jieba()
    {
        long total = 0;
        foreach (var line in ChineseText.ReadLines("jieba_dict.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            string word = line.Substring(0, tab);
            long frequency = long.Parse(line.Substring(tab + 1), CultureInfo.InvariantCulture);
            _frequency[word] = frequency;
            total += frequency;
            var codepoints = ChineseText.Codepoints(word);
            var prefix = new StringBuilder();
            for (int i = 0; i < codepoints.Count - 1; i++)
            {
                prefix.Append(codepoints[i]);
                string fragment = prefix.ToString();
                if (!_frequency.ContainsKey(fragment)) _frequency[fragment] = 0;
            }
        }
        _logTotal = Math.Log(total);

        var hmm = JObject.Parse(string.Join("\n", ChineseText.ReadLines("jieba_hmm.json.gz")));
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

        _tags = new Lazy<Dictionary<string, string>>(() =>
        {
            var tags = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var line in ChineseText.ReadLines("jieba_pos.tsv.gz"))
            {
                int tab = line.IndexOf('\t');
                tags[line.Substring(0, tab)] = line.Substring(tab + 1);
            }
            return tags;
        }, isThreadSafe: true);
        _posModel = new Lazy<PosModel>(() => new PosModel(JObject.Parse(string.Join("\n", ChineseText.ReadLines("jieba_posseg.json.gz")))), isThreadSafe: true);
    }

    private static readonly Regex JiebaHan = new("([一-鿕a-zA-Z0-9+#&\\._%\\-]+)");
    private static readonly Regex JiebaFastHan = new("([一-鿕a-zA-Z0-9+#&\\._%]+)");
    private static readonly Regex JiebaSkip = new("(\r\n|\\s)");
    private static readonly Regex HmmHan = new("([一-鿕]+)");
    private static readonly Regex HmmSkip = new("([a-zA-Z0-9]+(?:\\.\\d+)?%?)");

    /// <summary>jieba's <c>lcut(text, cut_all=False)</c> (HMM on), or jieba_fast's.</summary>
    public List<string> Cut(string sentence, bool jiebaFast = false)
    {
        var words = new List<string>();
        var han = jiebaFast ? JiebaFastHan : JiebaHan;
        foreach (var block in han.Split(sentence))
        {
            if (block.Length == 0) continue;
            if (han.IsMatch(block) && han.Match(block).Index == 0)
            {
                words.AddRange(CutDag(ChineseText.Codepoints(block), jiebaFast));
                continue;
            }
            foreach (var x in JiebaSkip.Split(block))
            {
                if (x.Length == 0) continue;
                if (JiebaSkip.Match(x) is { Success: true, Index: 0 }) words.Add(x);
                else words.AddRange(ChineseText.Codepoints(x));
            }
        }
        return words;
    }

    /// <summary>jieba's <c>lcut_for_search(text)</c>: each word of <see cref="Cut"/>, preceded by its dictionary 2-grams
    /// (words over 2 characters) and 3-grams (over 3).</summary>
    public List<string> CutForSearch(string sentence, bool jiebaFast = false)
    {
        var result = new List<string>();
        foreach (var word in Cut(sentence, jiebaFast))
        {
            var w = ChineseText.Codepoints(word);
            if (w.Count > 2)
                for (int i = 0; i < w.Count - 1; i++)
                {
                    string gram = w[i] + w[i + 1];
                    if (Frequency(gram) is > 0) result.Add(gram);
                }
            if (w.Count > 3)
                for (int i = 0; i < w.Count - 2; i++)
                {
                    string gram = w[i] + w[i + 1] + w[i + 2];
                    if (Frequency(gram) is > 0) result.Add(gram);
                }
            result.Add(word);
        }
        return result;
    }

    private long? Frequency(string word) => _frequency.TryGetValue(word, out long f) ? f : null;

    // Tokenizer.get_DAG and calc: the maximum-probability route over the dictionary's words. jieba_fast's
    // _get_DAG_and_calc keeps at most 12 words per position and takes the first (shortest) word on a tie.
    private (double Score, int End)[] Route(List<string> sentence, bool jiebaFast = false)
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
            while (i < n && _frequency.TryGetValue(fragment, out long f) && !(jiebaFast && ends.Count >= 12))
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
            (double Score, int End) best = jiebaFast ? (int.MinValue, 0) : (double.NegativeInfinity, -1);
            foreach (int x in dag[idx])
            {
                long f = Frequency(Slice(idx, x + 1)) ?? 0;
                double score = Math.Log(f == 0 ? 1 : f) - _logTotal + route[x + 1].Score;
                if (score > best.Score || (!jiebaFast && score == best.Score && x > best.End)) best = (score, x);
            }
            route[idx] = best;
        }
        return route;
    }

    // Tokenizer.__cut_DAG
    private IEnumerable<string> CutDag(List<string> sentence, bool jiebaFast)
    {
        int n = sentence.Count;
        string Slice(int from, int to) => string.Concat(sentence.GetRange(from, to - from));
        var route = Route(sentence, jiebaFast);
        var buffer = new StringBuilder();
        int bufferCount = 0;
        IEnumerable<string> Flush()
        {
            if (bufferCount == 0) yield break;
            string pending = buffer.ToString();
            if (bufferCount == 1) yield return pending;
            else if (Frequency(pending) is not > 0)
                foreach (var t in FinalSegCut(pending, jiebaFast)) yield return t;
            else
                foreach (var c in ChineseText.Codepoints(pending)) yield return c;
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
    private IEnumerable<string> FinalSegCut(string sentence, bool jiebaFast)
    {
        foreach (var block in HmmHan.Split(sentence))
        {
            if (block.Length == 0) continue;
            if (HmmHan.Match(block) is { Success: true, Index: 0 } m && m.Length == block.Length)
            {
                foreach (var word in Viterbi(ChineseText.Codepoints(block), jiebaFast)) yield return word;
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

    // finalseg.__cut: the B/M/E/S Viterbi path, then words from it. jieba_fast's C _viterbi adds emission, previous score
    // and transition in that order, keeps the first predecessor that strictly beats MIN_FLOAT (or, when none does,
    // the larger state letter with a MIN_FLOAT score), and ends on 'S' only when it strictly beats 'E'.
    private IEnumerable<string> Viterbi(List<string> observations, bool jiebaFast)
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
                if (jiebaFast)
                {
                    best = (MinFloat, '\0');
                    foreach (char y0 in PreviousStates[y])
                    {
                        double score = emission + probability[y0] + Transition(y0, y);
                        if (score > best.Score) best = (score, y0);
                    }
                    if (best.State == '\0') best = (MinFloat, PreviousStates[y].Max());
                }
                else
                {
                    foreach (char y0 in PreviousStates[y])
                    {
                        double score = probability[y0] + Transition(y0, y) + emission;
                        if (score > best.Score || (score == best.Score && y0 > best.State)) best = (score, y0);
                    }
                }
                next[y] = best.Score;
                nextPath[y] = new List<char>(path[best.State]) { y };
            }
            probability = next;
            path = nextPath;
        }
        // max((V[-1][y], y) for y in "ES"): a tie goes to "S".
        char last = jiebaFast
            ? (probability['S'] > probability['E'] ? 'S' : 'E')
            : (probability['E'] > probability['S'] ? 'E' : 'S');
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

    // ---------------------------------------------------------------- posseg

    private sealed class PosModel
    {
        public PosModel(JObject model)
        {
            foreach (var entry in (JObject)model["char_state"]!)
                CharStates[entry.Key] = ((JArray)entry.Value!).Select(v => (string)v!).ToList();
            foreach (var entry in (JObject)model["start"]!) Start[entry.Key] = (double)entry.Value!;
            foreach (var from in (JObject)model["trans"]!)
            {
                var row = new Dictionary<string, double>(StringComparer.Ordinal);
                foreach (var to in (JObject)from.Value!) row[to.Key] = (double)to.Value!;
                Transition[from.Key] = row;
            }
            foreach (var state in (JObject)model["emit"]!)
            {
                var row = new Dictionary<string, double>(StringComparer.Ordinal);
                foreach (var e in (JObject)state.Value!) row[e.Key] = (double)e.Value!;
                Emission[state.Key] = row;
            }
        }

        public Dictionary<string, List<string>> CharStates { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, double> Start { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, Dictionary<string, double>> Transition { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, Dictionary<string, double>> Emission { get; } = new(StringComparer.Ordinal);
    }

    private static readonly Regex PosHanInternal = new("([一-鿕a-zA-Z0-9+#&\\._]+)");
    private static readonly Regex PosSkipInternal = new("(\r\n|\\s)");
    private static readonly Regex PosHanDetail = new("([一-鿕]+)");
    private static readonly Regex PosSkipDetail = new("([\\.0-9]+|[a-zA-Z0-9]+)");
    private static readonly Regex PosEng = new("^[a-zA-Z0-9]+");
    private static readonly Regex PosNum = new("^[\\.0-9]+");

    private string Tag(string word) => _tags.Value.TryGetValue(word, out var tag) ? tag : "x";

    /// <summary>jieba's <c>posseg.lcut(text)</c> (HMM on): each word and its part-of-speech tag.</summary>
    public List<(string Word, string Tag)> PosCut(string sentence)
    {
        var result = new List<(string, string)>();
        foreach (var block in PosHanInternal.Split(sentence))
        {
            if (PosHanInternal.Match(block) is { Success: true, Index: 0 })
            {
                result.AddRange(PosCutDag(ChineseText.Codepoints(block)));
                continue;
            }
            foreach (var x in PosSkipInternal.Split(block))
            {
                if (PosSkipInternal.Match(x) is { Success: true, Index: 0 })
                {
                    result.Add((x, "x"));
                    continue;
                }
                foreach (var xx in ChineseText.Codepoints(x))
                {
                    // The reference tests the whole run x for English, not the character xx.
                    if (PosNum.IsMatch(xx)) result.Add((xx, "m"));
                    else if (PosEng.IsMatch(x)) result.Add((xx, "eng"));
                    else result.Add((xx, "x"));
                }
            }
        }
        return result;
    }

    // POSTokenizer.__cut_DAG
    private IEnumerable<(string, string)> PosCutDag(List<string> sentence)
    {
        int n = sentence.Count;
        var route = Route(sentence);
        var buffer = new List<string>();
        IEnumerable<(string, string)> Flush()
        {
            if (buffer.Count == 0) yield break;
            string pending = string.Concat(buffer);
            if (buffer.Count == 1) yield return (pending, Tag(pending));
            else if (Frequency(pending) is not > 0)
                foreach (var t in PosCutDetail(pending)) yield return t;
            else
                foreach (var c in buffer) yield return (c, Tag(c));
            buffer.Clear();
        }
        int at = 0;
        while (at < n)
        {
            int y = route[at].End + 1;
            string word = string.Concat(sentence.GetRange(at, y - at));
            if (y - at == 1)
            {
                buffer.Add(word);
            }
            else
            {
                foreach (var t in Flush()) yield return t;
                yield return (word, Tag(word));
            }
            at = y;
        }
        foreach (var t in Flush()) yield return t;
    }

    // POSTokenizer.__cut_detail
    private IEnumerable<(string, string)> PosCutDetail(string sentence)
    {
        foreach (var block in PosHanDetail.Split(sentence))
        {
            if (PosHanDetail.Match(block) is { Success: true, Index: 0 })
            {
                foreach (var t in PosViterbiCut(ChineseText.Codepoints(block))) yield return t;
                continue;
            }
            foreach (var x in PosSkipDetail.Split(block))
            {
                if (x.Length == 0) continue;
                if (PosNum.IsMatch(x)) yield return (x, "m");
                else if (PosEng.IsMatch(x)) yield return (x, "eng");
                else yield return (x, "x");
            }
        }
    }

    // posseg.viterbi over (position, tag) states, then words from the path (POSTokenizer.__cut).
    private IEnumerable<(string, string)> PosViterbiCut(List<string> obs)
    {
        var m = _posModel.Value;
        var allStates = m.Transition.Keys.ToList();
        IEnumerable<string> StatesOf(string c) => m.CharStates.TryGetValue(c, out var s) ? s : allStates;
        double Emit(string state, string c) => m.Emission[state].TryGetValue(c, out var p) ? p : MinFloat;
        var v = new List<Dictionary<string, double>> { new(StringComparer.Ordinal) };
        var memPath = new List<Dictionary<string, string>> { new(StringComparer.Ordinal) };
        foreach (var y in StatesOf(obs[0]))
        {
            v[0][y] = m.Start[y] + Emit(y, obs[0]);
            memPath[0][y] = "";
        }
        for (int t = 1; t < obs.Count; t++)
        {
            v.Add(new Dictionary<string, double>(StringComparer.Ordinal));
            memPath.Add(new Dictionary<string, string>(StringComparer.Ordinal));
            var prevStates = memPath[t - 1].Keys.Where(x => m.Transition[x].Count > 0).ToList();
            var expectNext = new HashSet<string>(prevStates.SelectMany(x => m.Transition[x].Keys), StringComparer.Ordinal);
            var obsStates = new HashSet<string>(StatesOf(obs[t]), StringComparer.Ordinal);
            obsStates.IntersectWith(expectNext);
            IEnumerable<string> candidates = obsStates.Count > 0 ? obsStates : expectNext.Count > 0 ? expectNext : allStates;
            foreach (var y in candidates)
            {
                (double Prob, string State) best = (double.NegativeInfinity, "");
                bool any = false;
                foreach (var y0 in prevStates)
                {
                    double trans = m.Transition[y0].TryGetValue(y, out var tp) ? tp : double.NegativeInfinity;
                    double prob = v[t - 1][y0] + trans + Emit(y, obs[t]);
                    // max over (prob, state) tuples: a tie goes to the greater state.
                    if (!any || prob > best.Prob || (prob == best.Prob && string.CompareOrdinal(y0, best.State) > 0)) best = (prob, y0);
                    any = true;
                }
                v[t][y] = best.Prob;
                memPath[t][y] = best.State;
            }
        }
        (double Prob, string State) last = (double.NegativeInfinity, "");
        bool first = true;
        foreach (var y in memPath[memPath.Count - 1].Keys)
        {
            double p = v[v.Count - 1][y];
            if (first || p > last.Prob || (p == last.Prob && string.CompareOrdinal(y, last.State) > 0)) last = (p, y);
            first = false;
        }
        var route = new string[obs.Count];
        string state = last.State;
        for (int i = obs.Count - 1; i >= 0; i--)
        {
            route[i] = state;
            state = memPath[i][state];
        }
        int begin = 0, nextIndex = 0;
        for (int i = 0; i < obs.Count; i++)
        {
            char position = route[i][0];
            string tag = route[i].Substring(2);
            if (position == 'B')
            {
                begin = i;
            }
            else if (position == 'E')
            {
                yield return (string.Concat(obs.GetRange(begin, i - begin + 1)), tag);
                nextIndex = i + 1;
            }
            else if (position == 'S')
            {
                yield return (obs[i], tag);
                nextIndex = i + 1;
            }
        }
        if (nextIndex < obs.Count) yield return (string.Concat(obs.GetRange(nextIndex, obs.Count - nextIndex)), route[nextIndex].Substring(2));
    }
}

/// <summary>
/// pypinyin 0.55.0's reading of a word (<c>lazy_pinyin</c> and <c>pinyin</c> with their defaults: Han and other runs, the
/// Han runs cut by maximum matching on the phrase dictionary, each piece read as a phrase or character by character, the
/// first reading only, a run without a reading kept as written), over readings exported in one style.
/// </summary>
internal sealed class PypinyinReader
{
    private static readonly Lazy<PypinyinReader> BopomofoInstance =
        new(() => new PypinyinReader("pypinyin_chars.tsv.gz", "pypinyin_phrases.tsv.gz"), isThreadSafe: true);

    private static readonly Lazy<PypinyinReader> EmotiVoiceInstance =
        new(() => new PypinyinReader("emotivoice_pinyin_chars.tsv.gz", "emotivoice_pinyin_phrases.tsv.gz"), isThreadSafe: true);

    private readonly Dictionary<string, string[]> _phrases = new(StringComparer.Ordinal);
    private readonly HashSet<string> _phrasePrefixes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _characters = new(StringComparer.Ordinal);

    /// <summary>pypinyin's BOPOMOFO style (VALL-E X's front end).</summary>
    public static PypinyinReader Bopomofo => BopomofoInstance.Value;

    /// <summary>pypinyin's TONE3 style with the neutral tone as 5, with pypinyin-dict's CC-CEDICT phrases loaded
    /// (EmotiVoice's front end; <c>tools/reference-data/emotivoice_frontend_reference.py</c>).</summary>
    public static PypinyinReader EmotiVoiceTone3 => EmotiVoiceInstance.Value;

    private PypinyinReader(string characters, string phrases)
    {
        foreach (var line in ChineseText.ReadLines(characters))
        {
            int tab = line.IndexOf('\t');
            _characters[line.Substring(0, tab)] = line.Substring(tab + 1);
        }
        foreach (var line in ChineseText.ReadLines(phrases))
        {
            int tab = line.IndexOf('\t');
            string phrase = line.Substring(0, tab);
            _phrases[phrase] = line.Substring(tab + 1).Split(' ');
            var codepoints = ChineseText.Codepoints(phrase);
            var prefix = new StringBuilder();
            foreach (var c in codepoints)
            {
                prefix.Append(c);
                _phrasePrefixes.Add(prefix.ToString());
            }
        }
    }

    // pypinyin's RE_HANS character ranges.
    private static readonly (int From, int To)[] HanRanges =
    {
        (0x3007, 0x3007), (0xE815, 0xE864), (0xFA18, 0xFA18), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF), (0x2A703, 0x2B73F), (0x2B740, 0x2B81D), (0x2B825, 0x2BF6E), (0x2C029, 0x2CE93),
        (0x2D016, 0x2D016), (0x2D11B, 0x2EBD9), (0x2F80A, 0x2FA1F), (0x30000, 0x3134A), (0x300F7, 0x31288),
        (0x30EDD, 0x30EDE), (0x31350, 0x32389),
    };

    internal static bool IsHan(string codepoint)
    {
        int value = char.ConvertToUtf32(codepoint, 0);
        foreach (var (from, to) in HanRanges)
            if (value >= from && value <= to) return true;
        return false;
    }

    // lazy_pinyin(word, BOPOMOFO): simple_seg into Han and other runs, the Han runs cut by forward maximum matching on
    // the phrase dictionary (no_non_phrases), each piece read as a phrase or character by character; a run without a
    // reading is kept as written.
    public IEnumerable<string> Read(string word)
    {
        var codepoints = ChineseText.Codepoints(word);
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
}
