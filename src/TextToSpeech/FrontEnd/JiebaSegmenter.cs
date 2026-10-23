using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// jieba 0.42.1 (MIT), the Chinese word segmenter: <c>lcut</c> (its prefix dictionary's maximum-probability route, with
/// the B/M/E/S HMM for unknown words), <c>cut_for_search</c> (plus the dictionary's 2- and 3-character sub-words), and
/// <c>jieba.posseg.lcut</c> (the same route, words tagged from the dictionary, unknown runs segmented and tagged by the
/// part-of-speech HMM). The dictionaries are jieba's own, exported by <c>tools/reference-data/mandarin_g2p_resources.py</c>
/// and <c>melo_chinese_resources.py</c>.
/// </summary>
internal sealed class JiebaSegmenter
{
    private static readonly Lazy<JiebaSegmenter> SharedInstance = new(() => new JiebaSegmenter(), isThreadSafe: true);

    private readonly Dictionary<string, long> _frequency = new(StringComparer.Ordinal);
    private readonly double _logTotal;
    private readonly Dictionary<char, double> _hmmStart = new();
    private readonly Dictionary<char, Dictionary<char, double>> _hmmTransition = new();
    private readonly Dictionary<char, Dictionary<string, double>> _hmmEmission = new();
    private readonly Lazy<Dictionary<string, string>> _tags;
    private readonly Lazy<PosModel> _posModel;

    /// <summary>The shared instance (the dictionaries load once, on first use).</summary>
    public static JiebaSegmenter Default => SharedInstance.Value;

    private JiebaSegmenter()
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

        _tags = new Lazy<Dictionary<string, string>>(() =>
        {
            var tags = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var line in ReadLines("jieba_pos.tsv.gz"))
            {
                int tab = line.IndexOf('\t');
                tags[line.Substring(0, tab)] = line.Substring(tab + 1);
            }
            return tags;
        }, isThreadSafe: true);
        _posModel = new Lazy<PosModel>(() => new PosModel(JObject.Parse(string.Join("\n", ReadLines("jieba_posseg.json.gz")))), isThreadSafe: true);
    }

    internal static IEnumerable<string> ReadLines(string resource)
    {
        var assembly = typeof(JiebaSegmenter).Assembly;
        string name = assembly.GetManifestResourceNames().Single(n => n.EndsWith(resource, StringComparison.Ordinal));
        using var stream = assembly.GetManifestResourceStream(name)!;
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var lines = new List<string>();
        for (string? line = reader.ReadLine(); line is not null; line = reader.ReadLine())
            if (line.Length > 0) lines.Add(line);
        return lines;
    }

    /// <summary>Python iterates strings by code point; supplementary characters are surrogate pairs here.</summary>
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

    private long? Frequency(string word) => _frequency.TryGetValue(word, out long f) ? f : null;

    // ---------------------------------------------------------------- lcut / cut_for_search

    private static readonly Regex JiebaHan = new("([一-鿕a-zA-Z0-9+#&\\._%\\-]+)");
    private static readonly Regex JiebaSkip = new("(\r\n|\\s)");
    private static readonly Regex HmmHan = new("([一-鿕]+)");
    private static readonly Regex HmmSkip = new("([a-zA-Z0-9]+(?:\\.\\d+)?%?)");

    /// <summary>jieba's <c>lcut(text, cut_all=False)</c> (HMM on).</summary>
    public List<string> Cut(string sentence)
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

    /// <summary>jieba's <c>lcut_for_search(text)</c>: each word of <see cref="Cut"/>, preceded by its dictionary 2-grams
    /// (words over 2 characters) and 3-grams (over 3).</summary>
    public List<string> CutForSearch(string sentence)
    {
        var result = new List<string>();
        foreach (var word in Cut(sentence))
        {
            var w = Codepoints(word);
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

    // Tokenizer.get_DAG and calc: the maximum-probability route over the dictionary's words.
    private (int End, double Score)[] Route(List<string> sentence)
    {
        int n = sentence.Count;
        string Slice(int from, int to) => string.Concat(sentence.GetRange(from, to - from));
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
        var route = new (int End, double Score)[n + 1];
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
            route[idx] = (best.End, best.Score);
        }
        return route;
    }

    // Tokenizer.__cut_DAG
    private IEnumerable<string> CutDag(List<string> sentence)
    {
        int n = sentence.Count;
        var route = Route(sentence);
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
                yield return string.Concat(sentence.GetRange(at, y - at));
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
                result.AddRange(PosCutDag(Codepoints(block)));
                continue;
            }
            foreach (var x in PosSkipInternal.Split(block))
            {
                if (PosSkipInternal.Match(x) is { Success: true, Index: 0 })
                {
                    result.Add((x, "x"));
                    continue;
                }
                foreach (var xx in Codepoints(x))
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
                foreach (var t in PosViterbiCut(Codepoints(block))) yield return t;
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
