using System.IO.Compression;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// WeTextProcessing 1.0.4.1's text normalization (<c>tn.processor.Processor.normalize</c>), which MegaTTS 3 runs on its
/// input: the text's bytes composed with the compiled tagger grammar and the shortest path's output taken (a token string
/// such as <c>cardinal { integer: "一百" }</c>), its tokens' fields reordered by the token parser, then that composed with
/// the verbalizer grammar the same way.
/// </summary>
/// <remarks>
/// The grammars are the wheel's compiled ones (<c>tn/zh_tn_*.fst</c>, <c>tn/en_tn_*.fst</c>), optimized as the
/// normalizers optimize them on load, embedded in OpenFst's binary vector format; the runtime reads them and finds the
/// shortest path in the tropical semiring (single-precision weights) over the composition with the input string.
/// </remarks>
internal sealed class WeTextNormalizer
{
    private readonly OpenFst _tagger, _verbalizer;
    private readonly IReadOnlyDictionary<string, string[]> _orders;

    private WeTextNormalizer(string prefix, IReadOnlyDictionary<string, string[]> orders)
    {
        _tagger = OpenFst.ReadResource($"AiDotNet.TextToSpeech.FrontEnd.wetext_{prefix}_tagger.fst.gz");
        _verbalizer = OpenFst.ReadResource($"AiDotNet.TextToSpeech.FrontEnd.wetext_{prefix}_verbalizer.fst.gz");
        _orders = orders;
    }

    // token_parser.TN_ORDERS and EN_TN_ORDERS.
    private static readonly Dictionary<string, string[]> TnOrders = new()
    {
        ["date"] = new[] { "year", "month", "day" },
        ["fraction"] = new[] { "denominator", "numerator" },
        ["measure"] = new[] { "denominator", "numerator", "value" },
        ["money"] = new[] { "value", "currency" },
        ["time"] = new[] { "noon", "hour", "minute", "second" },
    };

    private static readonly Dictionary<string, string[]> EnTnOrders = new()
    {
        ["date"] = new[] { "preserve_order", "text", "day", "month", "year" },
        ["money"] = new[] { "integer_part", "fractional_part", "quantity", "currency_maj" },
    };

    private static readonly Lazy<WeTextNormalizer> ChineseInstance = new(() => new WeTextNormalizer("zh_tn", TnOrders), isThreadSafe: true);
    private static readonly Lazy<WeTextNormalizer> EnglishInstance = new(() => new WeTextNormalizer("en_tn", EnTnOrders), isThreadSafe: true);

    /// <summary>The Chinese normalizer (<c>tn.chinese.normalizer.Normalizer</c>).</summary>
    public static WeTextNormalizer Chinese => ChineseInstance.Value;

    /// <summary>The English normalizer (<c>tn.english.normalizer.Normalizer</c>).</summary>
    public static WeTextNormalizer English => EnglishInstance.Value;

    /// <summary><c>normalize</c>: tag, then verbalize.</summary>
    public string Normalize(string text) => Verbalize(Tag(text));

    /// <summary><c>tag</c>: the tagger's best token string for the text ("" for "").</summary>
    /// <exception cref="InvalidOperationException">The grammar accepts no path for the text.</exception>
    public string Tag(string text) => text.Length == 0 ? "" : _tagger.ShortestOutput(Encoding.UTF8.GetBytes(text));

    /// <summary><c>verbalize</c>: the token string reordered, then the verbalizer's best output ("" for "").</summary>
    public string Verbalize(string tagged) => tagged.Length == 0 ? "" : _verbalizer.ShortestOutput(Encoding.UTF8.GetBytes(Reorder(tagged)));

    // ------------------------------------------------------------------ token_parser.TokenParser.reorder

    private sealed class Token
    {
        public string Name = "";
        public List<string> Order = new();
        public Dictionary<string, string> Members = new(StringComparer.Ordinal);
    }

    /// <summary><c>TokenParser.reorder</c>: each token's fields in its class's order (unless it says
    /// <c>preserve_order: "true"</c>), tokens joined by spaces.</summary>
    internal string Reorder(string input)
    {
        var tokens = Parse(input);
        var output = new StringBuilder();
        foreach (var token in tokens)
        {
            var order = token.Order;
            if (_orders.TryGetValue(token.Name, out var classOrder)
                && (!token.Members.TryGetValue("preserve_order", out var preserve) || preserve != "true"))
                order = classOrder.ToList();
            output.Append(token.Name).Append(" {");
            foreach (var key in order)
                if (token.Members.TryGetValue(key, out var value)) output.Append(' ').Append(key).Append(": \"").Append(value).Append('"');
            output.Append(" } ");
        }
        return output.ToString().Trim(' ', '\t', '\n', '\r', '\v', '\f');
    }

    private static List<Token> Parse(string text)
    {
        int index = 0;
        bool eos = false;
        char ch = text[0];
        bool Read()
        {
            if (index < text.Length - 1)
            {
                ch = text[++index];
                return true;
            }
            eos = true;
            return false;
        }
        bool ParseWhitespace()
        {
            bool notEos = !eos;
            while (notEos && ch == ' ') notEos = Read();
            return notEos;
        }
        bool ParseChar(char expected)
        {
            if (eos || ch != expected) return false;
            Read();
            return true;
        }
        void ParseChars(string expected)
        {
            foreach (char c in expected) ParseChar(c);
        }
        string ParseKey()
        {
            var key = new StringBuilder();
            while (!eos && ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'))
            {
                key.Append(ch);
                Read();
            }
            return key.ToString();
        }
        string ParseValue()
        {
            var value = new StringBuilder();
            while (eos || ch != '"')
            {
                if (eos) throw new InvalidOperationException("Unterminated value in the token string.");
                value.Append(ch);
                bool escape = ch == '\\';
                Read();
                if (escape)
                {
                    if (eos) throw new InvalidOperationException("Unterminated value in the token string.");
                    value.Append(ch);
                    Read();
                }
            }
            return value.ToString();
        }
        var tokens = new List<Token>();
        while (ParseWhitespace())
        {
            var token = new Token { Name = ParseKey() };
            ParseChars(" { ");
            while (ParseWhitespace())
            {
                if (ch == '}')
                {
                    ParseChar('}');
                    break;
                }
                string key = ParseKey();
                ParseChars(": \"");
                string value = ParseValue();
                ParseChar('"');
                token.Order.Add(key);
                token.Members[key] = value;
            }
            tokens.Add(token);
        }
        return tokens;
    }
}

/// <summary>
/// An OpenFst <c>VectorFst&lt;StdArc&gt;</c> read from its binary form (header, then each state's final weight and arcs),
/// with the one operation text normalization needs: the output of the shortest path through its composition with a
/// byte string, in the tropical semiring with single-precision weights.
/// </summary>
internal sealed class OpenFst
{
    private const int Magic = 2125659606;
    private readonly int _start;
    private readonly float[] _final;
    private readonly int[] _arcStart;
    private readonly int[] _ilabel, _olabel, _next;
    private readonly float[] _weight;

    private OpenFst(int start, float[] final, int[] arcStart, int[] ilabel, int[] olabel, float[] weight, int[] next)
    {
        _start = start;
        _final = final;
        _arcStart = arcStart;
        _ilabel = ilabel;
        _olabel = olabel;
        _weight = weight;
        _next = next;
    }

    public static OpenFst ReadResource(string name)
    {
        using var stream = typeof(OpenFst).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"The grammar {name} is not embedded.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var buffer = new MemoryStream();
        gzip.CopyTo(buffer);
        buffer.Position = 0;
        return Read(buffer);
    }

    /// <summary>Reads a <c>vector</c> FST of <c>standard</c> arcs without symbol tables.</summary>
    public static OpenFst Read(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (reader.ReadInt32() != Magic) throw new InvalidDataException("Not an OpenFst binary.");
        string ReadString() => Encoding.ASCII.GetString(reader.ReadBytes(reader.ReadInt32()));
        string type = ReadString(), arcType = ReadString();
        if (type != "vector" || arcType != "standard") throw new InvalidDataException($"Unsupported FST {type}/{arcType}.");
        reader.ReadInt32();                       // version
        int flags = reader.ReadInt32();
        if ((flags & 3) != 0) throw new InvalidDataException("FSTs with symbol tables are not supported.");
        reader.ReadUInt64();                      // properties
        long start = reader.ReadInt64(), numStates = reader.ReadInt64();
        reader.ReadInt64();                       // arcs (not counted by vector FSTs)
        var final = new float[numStates];
        var arcStart = new int[numStates + 1];
        var ilabel = new List<int>();
        var olabel = new List<int>();
        var weight = new List<float>();
        var next = new List<int>();
        for (long s = 0; s < numStates; s++)
        {
            final[s] = reader.ReadSingle();
            long arcs = reader.ReadInt64();
            arcStart[s] = ilabel.Count;
            for (long a = 0; a < arcs; a++)
            {
                ilabel.Add(reader.ReadInt32());
                olabel.Add(reader.ReadInt32());
                weight.Add(reader.ReadSingle());
                next.Add(reader.ReadInt32());
            }
        }
        arcStart[numStates] = ilabel.Count;
        return new OpenFst((int)start, final, arcStart, ilabel.ToArray(), olabel.ToArray(), weight.ToArray(), next.ToArray());
    }

    /// <summary>
    /// <c>shortestpath(bytes @ fst, nshortest=1, unique=True).string()</c>: the output bytes, as UTF-8, of the least-weight
    /// accepting path whose input labels (epsilon 0 skipped) spell <paramref name="input"/>, ties broken as OpenFst breaks
    /// them.
    /// </summary>
    /// <remarks>
    /// The composition is built as OpenFst's (states (position, state), each state's arcs the grammar's in order:
    /// input-epsilon arcs, then those reading the next byte) and trimmed to the states on accepting paths. Its acyclicity is
    /// not a known property, so <c>AutoQueue</c> decomposes it into strongly connected components: with every arc weight
    /// 0 or infinite it takes a LIFO queue, otherwise (all components trivial, the grammars having no input-epsilon cycles)
    /// the topological order of a depth-first search. <c>SingleShortestPath</c> then relaxes arcs in queue order, keeping a
    /// state's first parent among equal distances, and the first final state reached among equal totals.
    /// </remarks>
    /// <exception cref="InvalidOperationException">No path accepts the input.</exception>
    public string ShortestOutput(byte[] input)
    {
        int n = input.Length;
        // Compose: the reachable states and their arcs.
        var ids = new Dictionary<long, int>();
        var position = new List<int>();
        var grammarState = new List<int>();
        var arcs = new List<List<(int Next, int GrammarArc)>>();
        int Find(int pos, int state)
        {
            long key = (long)pos << 32 | (uint)state;
            if (ids.TryGetValue(key, out int id)) return id;
            id = position.Count;
            ids[key] = id;
            position.Add(pos);
            grammarState.Add(state);
            arcs.Add(null!);
            return id;
        }
        Find(0, _start);
        for (int s = 0; s < position.Count; s++)
        {
            int pos = position[s], t = grammarState[s];
            var list = new List<(int, int)>();
            for (int a = _arcStart[t]; a < _arcStart[t + 1]; a++)
            {
                if (_ilabel[a] == 0) list.Add((Find(pos, _next[a]), a));
                else if (pos < n && _ilabel[a] == input[pos]) list.Add((Find(pos + 1, _next[a]), a));
            }
            arcs[s] = list;
        }
        int count = position.Count;
        float Final(int s) => position[s] == n ? _final[grammarState[s]] : float.PositiveInfinity;

        // Connect: keep the states that reach a final state.
        var predecessors = new List<int>[count];
        for (int s = 0; s < count; s++) predecessors[s] = new List<int>();
        for (int s = 0; s < count; s++)
            foreach (var (next, _) in arcs[s]) predecessors[next].Add(s);
        var live = new bool[count];
        var pending = new Stack<int>();
        for (int s = 0; s < count; s++)
            if (!float.IsPositiveInfinity(Final(s))) { live[s] = true; pending.Push(s); }
        while (pending.Count > 0)
            foreach (int p in predecessors[pending.Pop()])
                if (!live[p]) { live[p] = true; pending.Push(p); }
        if (!live[0]) throw new InvalidOperationException("The grammar accepts no path for the input.");
        for (int s = 0; s < count; s++) arcs[s] = live[s] ? arcs[s].Where(e => live[e.Next]).ToList() : new List<(int, int)>();

        // AutoQueue: LIFO when unweighted, otherwise the topological order of a depth-first search.
        bool unweighted = true;
        for (int s = 0; s < count && unweighted; s++)
            foreach (var (_, a) in arcs[s])
                if (_weight[a] != 0f && !float.IsPositiveInfinity(_weight[a])) { unweighted = false; break; }
        int[] order = Array.Empty<int>();
        if (!unweighted)
        {
            var finish = new List<int>();
            var color = new byte[count];
            var stack = new Stack<(int State, int Arc)>();
            stack.Push((0, 0));
            color[0] = 1;
            while (stack.Count > 0)
            {
                var (s, i) = stack.Pop();
                if (i == arcs[s].Count)
                {
                    color[s] = 2;
                    finish.Add(s);
                    continue;
                }
                int next = arcs[s][i].Next;
                if (color[next] == 0)
                {
                    stack.Push((s, i));
                    color[next] = 1;
                    stack.Push((next, 0));
                }
                else
                {
                    if (color[next] == 1) throw new InvalidOperationException("The composition has a cycle.");
                    stack.Push((s, i + 1));
                }
            }
            order = new int[count];
            for (int k = 0; k < finish.Count; k++) order[finish[finish.Count - 1 - k]] = k;
        }

        // SingleShortestPath.
        var distance = new float[count];
        var parent = new (int State, int Arc)[count];
        var enqueued = new bool[count];
        for (int s = 0; s < count; s++) { distance[s] = float.PositiveInfinity; parent[s] = (-1, -1); }
        distance[0] = 0f;
        var lifo = new Stack<int>();
        var topState = new int[unweighted ? 0 : count];
        for (int k = 0; k < topState.Length; k++) topState[k] = -1;
        int front = 0, back = -1;
        void Enqueue(int s)
        {
            enqueued[s] = true;
            if (unweighted) { lifo.Push(s); return; }
            int k = order[s];
            if (front > back) front = back = k;
            else if (k > back) back = k;
            else if (k < front) front = k;
            topState[k] = s;
        }
        bool Empty() => unweighted ? lifo.Count == 0 : front > back;
        int Dequeue()
        {
            if (unweighted) return lifo.Pop();
            int s = topState[front];
            topState[front] = -1;
            while (front <= back && topState[front] == -1) ++front;
            return s;
        }
        Enqueue(0);
        float finalDistance = float.PositiveInfinity;
        int finalParent = -1;
        while (!Empty())
        {
            int s = Dequeue();
            enqueued[s] = false;
            float sd = distance[s];
            float f = Final(s);
            if (!float.IsPositiveInfinity(f))
            {
                float total = sd + f;
                if (total < finalDistance) { finalDistance = total; finalParent = s; }
            }
            for (int i = 0; i < arcs[s].Count; i++)
            {
                var (next, a) = arcs[s][i];
                float w = sd + _weight[a];
                if (!(w < distance[next])) continue;
                distance[next] = w;
                parent[next] = (s, a);
                if (!enqueued[next]) Enqueue(next);
            }
        }
        if (finalParent < 0) throw new InvalidOperationException("The grammar accepts no path for the input.");

        var output = new List<byte>();
        for (int s = finalParent; parent[s].State >= 0; s = parent[s].State)
        {
            int a = parent[s].Arc;
            if (_olabel[a] != 0) output.Add((byte)_olabel[a]);
        }
        output.Reverse();
        return Encoding.UTF8.GetString(output.ToArray());
    }
}
