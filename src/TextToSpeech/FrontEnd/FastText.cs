using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// fastText's supervised prediction (Facebook; fasttext-predict 0.9.2.4, MIT) for a dense model file
/// (<c>lid.176.bin</c>, the language identifier fast_langdetect loads as its "full" model): the line split into words,
/// each word's id and character n-grams hashed into buckets, the hidden vector the mean of their input rows, and the
/// k best labels by the loss (hierarchical softmax, softmax or one-vs-all), in float32 as the C++ computes them.
/// </summary>
internal sealed class FastText
{
    private const int MagicInt32 = 793712314;
    private const int MaxVersion = 12;

    private readonly int _dim, _wordNgrams, _bucket, _minn, _maxn, _loss;
    private readonly List<(byte[] Word, long Count, byte Type)> _entries = new();
    private readonly int _nwords, _nlabels;
    private readonly long _pruneSize;
    private readonly Dictionary<int, int> _prune = new();
    private readonly int[] _word2int;
    private readonly List<int>[] _subwords;
    private readonly float[] _input;
    private readonly float[] _output;
    private readonly int _outputRows;
    // Hierarchical softmax tree.
    private readonly (int Parent, int Left, int Right, long Count, bool Binary)[] _tree = Array.Empty<(int, int, int, long, bool)>();

    public FastText(string path)
    {
        using var reader = new BinaryReader(File.OpenRead(path));
        if (reader.ReadInt32() != MagicInt32) throw new InvalidDataException($"{path} has wrong file format!");
        int version = reader.ReadInt32();
        if (version > MaxVersion) throw new InvalidDataException($"{path} has wrong file format!");
        _dim = reader.ReadInt32();
        reader.ReadInt32();                // ws
        reader.ReadInt32();                // epoch
        reader.ReadInt32();                // minCount
        reader.ReadInt32();                // neg
        _wordNgrams = reader.ReadInt32();
        _loss = reader.ReadInt32();
        int model = reader.ReadInt32();
        _bucket = reader.ReadInt32();
        _minn = reader.ReadInt32();
        _maxn = reader.ReadInt32();
        reader.ReadInt32();                // lrUpdateRate
        reader.ReadDouble();               // t
        if (model != 3) throw new InvalidDataException("Model needs to be supervised for prediction!");
        if (version == 11) _maxn = 0;

        int size = reader.ReadInt32();
        _nwords = reader.ReadInt32();
        _nlabels = reader.ReadInt32();
        reader.ReadInt64();                // ntokens
        _pruneSize = reader.ReadInt64();
        for (int i = 0; i < size; i++)
        {
            var word = new List<byte>();
            for (byte c = reader.ReadByte(); c != 0; c = reader.ReadByte()) word.Add(c);
            long count = reader.ReadInt64();
            byte type = reader.ReadByte();
            _entries.Add((word.ToArray(), count, type));
        }
        for (long i = 0; i < _pruneSize; i++) _prune[reader.ReadInt32()] = reader.ReadInt32();
        _subwords = new List<int>[size];
        for (int i = 0; i < size; i++)
        {
            _subwords[i] = new List<int> { i };
            if (!Encoding.UTF8.GetString(_entries[i].Word).Equals(Eos, StringComparison.Ordinal))
                ComputeSubwords(Wrap(_entries[i].Word), _subwords[i]);
        }
        _word2int = new int[(int)Math.Ceiling(size / 0.7)];
        for (int i = 0; i < _word2int.Length; i++) _word2int[i] = -1;
        for (int i = 0; i < size; i++) _word2int[Find(_entries[i].Word, Hash(_entries[i].Word))] = i;

        if (reader.ReadBoolean()) throw new NotSupportedException("Quantized fastText models (.ftz) are not supported; use the dense .bin model.");
        _input = ReadMatrix(reader, out _, out int inputColumns);
        if (inputColumns != _dim) throw new InvalidDataException("The input matrix does not match the dimension.");
        reader.ReadBoolean();              // qout
        _output = ReadMatrix(reader, out _outputRows, out _);
        if (_loss == 1) _tree = BuildTree(_entries.Skip(_nwords).Select(e => e.Count).ToList());
        else if (_loss == 2) throw new NotSupportedException("A negative-sampling model predicts as one-vs-all only in training; it has no supervised prediction here.");
    }

    private const string Eos = "</s>";

    private static float[] ReadMatrix(BinaryReader reader, out int rows, out int columns)
    {
        long m = reader.ReadInt64(), n = reader.ReadInt64();
        rows = (int)m;
        columns = (int)n;
        var bytes = reader.ReadBytes(checked((int)(m * n * 4)));
        var data = new float[m * n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return data;
    }

    // Dictionary::hash: FNV-1a over the bytes read as signed chars.
    private static uint Hash(byte[] word)
    {
        uint h = 2166136261;
        foreach (byte b in word)
        {
            h ^= unchecked((uint)(sbyte)b);
            h = unchecked(h * 16777619);
        }
        return h;
    }

    private int Find(byte[] word, uint h)
    {
        int size = _word2int.Length;
        int id = (int)(h % (uint)size);
        while (_word2int[id] != -1 && !_entries[_word2int[id]].Word.AsSpan().SequenceEqual(word)) id = (id + 1) % size;
        return id;
    }

    private static byte[] Wrap(byte[] word)
    {
        var wrapped = new byte[word.Length + 2];
        wrapped[0] = (byte)'<';
        Array.Copy(word, 0, wrapped, 1, word.Length);
        wrapped[wrapped.Length - 1] = (byte)'>';
        return wrapped;
    }

    private void PushHash(List<int> hashes, int id)
    {
        if (_pruneSize == 0 || id < 0) return;
        if (_pruneSize > 0)
        {
            if (!_prune.TryGetValue(id, out id)) return;
        }
        hashes.Add(_nwords + id);
    }

    // Dictionary::computeSubwords: every n-gram of minn..maxn UTF-8 characters, except a single character at either end.
    private void ComputeSubwords(byte[] word, List<int> ngrams)
    {
        for (int i = 0; i < word.Length; i++)
        {
            if ((word[i] & 0xC0) == 0x80) continue;
            var ngram = new List<byte>();
            for (int j = i, n = 1; j < word.Length && n <= _maxn; n++)
            {
                ngram.Add(word[j++]);
                while (j < word.Length && (word[j] & 0xC0) == 0x80) ngram.Add(word[j++]);
                if (n >= _minn && !(n == 1 && (i == 0 || j == word.Length)))
                    PushHash(ngrams, (int)(Hash(ngram.ToArray()) % (uint)_bucket));
            }
        }
    }

    private static (int, int, int, long, bool)[] BuildTree(List<long> counts)
    {
        int osz = counts.Count;
        var tree = new (int Parent, int Left, int Right, long Count, bool Binary)[2 * osz - 1];
        for (int i = 0; i < tree.Length; i++) tree[i] = (-1, -1, -1, (long)1e15, false);
        for (int i = 0; i < osz; i++) tree[i].Count = counts[i];
        int leaf = osz - 1, node = osz;
        for (int i = osz; i < 2 * osz - 1; i++)
        {
            var mini = new int[2];
            for (int j = 0; j < 2; j++)
            {
                if (leaf >= 0 && tree[leaf].Count < tree[node].Count) mini[j] = leaf--;
                else mini[j] = node++;
            }
            tree[i].Left = mini[0];
            tree[i].Right = mini[1];
            tree[i].Count = tree[mini[0]].Count + tree[mini[1]].Count;
            tree[mini[0]].Parent = i;
            tree[mini[1]].Parent = i;
            tree[mini[1]].Binary = true;
        }
        return tree.Select(t => (t.Parent, t.Left, t.Right, t.Count, t.Binary)).ToArray();
    }

    // Dictionary::getLine on "text\n": words split at ' ' \n \r \t \v \f \0, "</s>" at the newline; each word's
    // subwords (in vocabulary) or n-grams (out of it), labels skipped; then word n-grams.
    private List<int> Words(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text + "\n");
        var words = new List<int>();
        var hashes = new List<int>();
        int at = 0;
        while (true)
        {
            var token = new List<byte>();
            bool eos = false, any = false;
            while (at < bytes.Length)
            {
                byte c = bytes[at++];
                if (c is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t' or 0x0B or 0x0C or 0)
                {
                    if (token.Count == 0)
                    {
                        if (c == (byte)'\n')
                        {
                            eos = true;
                            any = true;
                            break;
                        }
                        continue;
                    }
                    if (c == (byte)'\n') at--;
                    any = true;
                    break;
                }
                token.Add(c);
            }
            if (!any && token.Count == 0) break;
            byte[] word = eos ? Encoding.ASCII.GetBytes(Eos) : token.ToArray();
            uint h = Hash(word);
            int wid = _word2int[Find(word, h)];
            bool label = wid < 0 ? Encoding.UTF8.GetString(word).StartsWith("__label__", StringComparison.Ordinal) : _entries[wid].Type == 1;
            if (!label)
            {
                if (wid < 0)
                {
                    if (!eos) ComputeSubwords(Wrap(word), words);
                }
                else if (_maxn <= 0) words.Add(wid);
                else words.AddRange(_subwords[wid]);
                hashes.Add(unchecked((int)h));
            }
            if (eos) break;
            if (at >= bytes.Length) break;
        }
        for (int i = 0; i < hashes.Count; i++)
        {
            ulong h = unchecked((ulong)hashes[i]);
            for (int j = i + 1; j < hashes.Count && j < i + _wordNgrams; j++)
            {
                h = unchecked(h * 116049371 + (ulong)hashes[j]);
                PushHash(words, (int)(h % (ulong)_bucket));
            }
        }
        return words;
    }

    private static float StdLog(float x) => (float)Math.Log(x + 1e-5);

    // libstdc++'s heap with comparePairs (l.first > r.first): a min-heap on the score, the order ties come out in.
    private static bool Comp((float Score, int Id) l, (float Score, int Id) r) => l.Score > r.Score;

    private static void PushHeapAt(List<(float, int)> heap, int hole, int top, (float, int) value)
    {
        int parent = (hole - 1) / 2;
        while (hole > top && Comp(heap[parent], value))
        {
            heap[hole] = heap[parent];
            hole = parent;
            parent = (hole - 1) / 2;
        }
        heap[hole] = value;
    }

    private static void AdjustHeap(List<(float, int)> heap, int hole, int length, (float, int) value)
    {
        int top = hole, second = hole;
        while (second < (length - 1) / 2)
        {
            second = 2 * (second + 1);
            if (Comp(heap[second], heap[second - 1])) second--;
            heap[hole] = heap[second];
            hole = second;
        }
        if ((length & 1) == 0 && second == (length - 2) / 2)
        {
            second = 2 * (second + 1);
            heap[hole] = heap[second - 1];
            hole = second - 1;
        }
        PushHeapAt(heap, hole, top, value);
    }

    private static void PushHeap(List<(float, int)> heap) => PushHeapAt(heap, heap.Count - 1, 0, heap[heap.Count - 1]);

    private static void PopHeap(List<(float, int)> heap, int last)
    {
        if (last <= 1) return;
        var value = heap[last - 1];
        heap[last - 1] = heap[0];
        AdjustHeap(heap, 0, last - 1, value);
    }

    private static void SortHeap(List<(float, int)> heap)
    {
        for (int last = heap.Count; last > 1; last--) PopHeap(heap, last);
    }

    private static void Offer(List<(float, int)> heap, int k, (float, int) item)
    {
        heap.Add(item);
        PushHeap(heap);
        if (heap.Count > k)
        {
            PopHeap(heap, heap.Count);
            heap.RemoveAt(heap.Count - 1);
        }
    }

    private float DotRow(float[] hidden, int row)
    {
        float d = 0;
        for (int j = 0; j < _dim; j++) d += _output[row * _dim + j] * hidden[j];
        if (float.IsNaN(d)) throw new ArithmeticException("Encountered NaN.");
        return d;
    }

    /// <summary><c>model.predict(text, k, threshold)</c>: the labels (without "__label__") and probabilities.</summary>
    public List<(string Label, double Probability)> Predict(string text, int k = 1, float threshold = 0)
    {
        if (text.Contains('\n')) throw new ArgumentException("predict processes one line at a time (remove '\\n')", nameof(text));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "k needs to be 1 or higher!");
        var input = Words(text);
        var result = new List<(string, double)>();
        if (input.Count == 0) return result;
        var hidden = new float[_dim];
        foreach (int row in input)
            for (int j = 0; j < _dim; j++) hidden[j] += _input[row * _dim + j];
        float scale = (float)(1.0 / input.Count);
        for (int j = 0; j < _dim; j++) hidden[j] *= scale;

        var heap = new List<(float, int)>(k + 1);
        if (_loss == 1)
        {
            int osz = _nlabels;
            void Dfs(int node, float score)
            {
                if (score < StdLog(threshold)) return;
                if (heap.Count == k && score < heap[0].Item1) return;
                if (_tree[node].Left == -1 && _tree[node].Right == -1)
                {
                    Offer(heap, k, (score, node));
                    return;
                }
                float f = DotRow(hidden, node - osz);
                f = (float)(1.0 / (1 + MathF.Exp(-f)));
                Dfs(_tree[node].Left, score + StdLog(1.0f - f));
                Dfs(_tree[node].Right, score + StdLog(f));
            }
            Dfs(2 * osz - 2, 0f);
        }
        else
        {
            var output = new float[_outputRows];
            for (int i = 0; i < _outputRows; i++) output[i] = DotRow(hidden, i);
            if (_loss == 3)
            {
                float max = output[0], z = 0;
                for (int i = 0; i < output.Length; i++) max = Math.Max(output[i], max);
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = MathF.Exp(output[i] - max);
                    z += output[i];
                }
                for (int i = 0; i < output.Length; i++) output[i] /= z;
            }
            else
            {
                for (int i = 0; i < output.Length; i++) output[i] = TableSigmoid(output[i]);
            }
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] < threshold) continue;
                if (heap.Count == k && StdLog(output[i]) < heap[0].Item1) continue;
                Offer(heap, k, (StdLog(output[i]), i));
            }
        }
        SortHeap(heap);
        foreach (var (score, id) in heap)
            result.Add((Encoding.UTF8.GetString(_entries[id + _nwords].Word).Replace("__label__", ""), MathF.Exp(score)));
        return result;
    }

    // Loss::sigmoid's 512-entry table over [-8, 8].
    private static readonly float[] SigmoidTable = Enumerable.Range(0, 513)
        .Select(i => (float)(1.0 / (1.0 + Math.Exp(-((float)(i * 2 * 8) / 512 - 8))))).ToArray();

    private static float TableSigmoid(float x)
    {
        if (x < -8) return 0f;
        if (x > 8) return 1f;
        return SigmoidTable[(long)((x + 8) * 512 / 8 / 2)];
    }
}

/// <summary>
/// fast_langdetect 1.1.0's <c>detect(text, model="full", k, threshold)</c> (LlmKira; MIT) over fastText's lid.176.bin:
/// newlines made spaces, the text cut to 80 characters, lower cased when it is upper case (or over 80% capitals and
/// longer than 5), then the k best languages.
/// </summary>
internal sealed class FastLangDetect
{
    private readonly FastText _model;

    /// <summary>Loads <c>lid.176.bin</c> (fastText's language identification model, CC BY-SA 3.0).</summary>
    public FastLangDetect(string modelPath) => _model = new FastText(modelPath);

    private static readonly Regex Capital = new("[A-Z]");
    private static readonly Regex Letter = new("[A-Za-z]");

    /// <summary>The detected languages, most probable first.</summary>
    public List<(string Lang, double Score)> Detect(string text, int k = 1, float threshold = 0)
    {
        text = text.Replace("\n", " ");
        var codepoints = ChineseText.Codepoints(text);
        if (codepoints.Count > 80) text = string.Concat(codepoints.Take(80));
        if (PythonText.IsUpper(text) || (Capital.Matches(text).Count > 0.8 * Letter.Matches(text).Count && ChineseText.Codepoints(text).Count > 5))
            text = PythonText.Lower(text);
        return _model.Predict(text, k, threshold).Select(p => (p.Label, Math.Min(p.Probability, 1.0)))
            .OrderByDescending(p => p.Item2).ToList();
    }
}
