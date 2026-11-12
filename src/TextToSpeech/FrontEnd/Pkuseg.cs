using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// spacy-pkuseg 1.0.2's Chinese word segmenter as <c>pkuseg()</c> builds it (the <c>spacy_ontonotes</c> CRF, no user
/// dictionary, the bundled <c>default.pkl</c> word list as its post-processor): <see cref="Cut"/> is <c>pkuseg.cut</c>.
/// </summary>
/// <remarks>
/// <para>Each whitespace-separated fragment is labelled character by character by a linear-chain CRF over pkuseg's
/// unigram/bigram and dictionary-word features (characters normalized: <c>-._,|/*:</c> to <c>&amp;</c>, digits and
/// full-width letters to classes), decoded by its backward Viterbi; a character tagged with a B starts a word. Adjacent
/// words that together form a dictionary word of two to seven pieces are merged unless every piece is itself one.</para>
/// <para>The model (<c>features.msgpack</c>, <c>weights.npz</c>) is the release's download: <see cref="DefaultModelDirectory"/>
/// reuses pkuseg's own cache (<c>PKUSEG_HOME</c>, else <c>~/.pkuseg</c>) or downloads and checks it.</para>
/// </remarks>
internal sealed class Pkuseg
{
    /// <summary>The model the release's <c>pkuseg()</c> loads.</summary>
    public const string DefaultModel = "spacy_ontonotes";
    private const string ModelUrl = "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip";
    private const string ModelSha256 = "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe";
    private const int WordMax = 6, WordMin = 2;
    private const string NoWord = "**noWord";
    private static readonly HashSet<string> Numbers = new(ChineseText.Codepoints("0123456789.\u51E0\u4E8C\u4E09\u56DB\u4E94\u516D\u4E03\u516B\u4E5D\u5341\u5343\u4E07\u4EBF\u5146\u96F6\uFF11\uFF12\uFF13\uFF14\uFF15\uFF16\uFF17\uFF18\uFF19\uFF10\uFF05"));
    private static readonly HashSet<string> Letters = new(ChineseText.Codepoints(
        "\uFF21\uFF22\uFF23\uFF24\uFF25\uFF26\uFF27\uFF28\uFF29\uFF2A\uFF2B\uFF2C\uFF2D\uFF2E\uFF2F\uFF30\uFF31\uFF32\uFF33\uFF34\uFF35\uFF36\uFF37\uFF38\uFF39\uFF3A\uFF41\uFF42\uFF43\uFF44\uFF45\uFF46\uFF47\uFF48\uFF49\uFF47\uFF4B\uFF4C\uFF4D\uFF4E\uFF4F\uFF50\uFF51\uFF52\uFF53\uFF54\uFF55\uFF56\uFF57\uFF58\uFF59\uFF5A\uFF0F\u30FB\uFF0D"));
    private static readonly Lazy<HashSet<string>> DictionaryWords = new(LoadDictionaryWords, isThreadSafe: true);

    private readonly HashSet<string> _unigram;
    private readonly Dictionary<string, int> _features;
    private readonly bool[] _beginsWord;
    private readonly int _tags, _featureCount;
    private readonly double[] _weights;

    private Pkuseg(HashSet<string> unigram, Dictionary<string, int> features, Dictionary<string, int> tags, int featureCount, double[] weights)
    {
        _unigram = unigram;
        _features = features;
        _tags = tags.Count;
        _featureCount = featureCount;
        _weights = weights;
        _beginsWord = new bool[_tags];
        foreach (var pair in tags) _beginsWord[pair.Value] = pair.Key.IndexOf('B') >= 0;
        if (weights.Length != _tags * (featureCount + _tags))
            throw new InvalidDataException("pkuseg's weights do not match its feature and tag counts.");
    }

    /// <summary>Loads a model directory (<c>features.msgpack</c> and <c>weights.npz</c>).</summary>
    public static Pkuseg Load(string modelDirectory)
    {
        var data = (Dictionary<object, object?>)MessagePackReader.Read(File.ReadAllBytes(Path.Combine(modelDirectory, "features.msgpack")))!;
        var unigram = new HashSet<string>(((List<object?>)data["unigram"]!).Cast<string>(), StringComparer.Ordinal);
        var features = ((Dictionary<object, object?>)data["feature_to_idx"]!).ToDictionary(p => (string)p.Key, p => Convert.ToInt32(p.Value), StringComparer.Ordinal);
        var tags = ((Dictionary<object, object?>)data["tag_to_idx"]!).ToDictionary(p => (string)p.Key, p => Convert.ToInt32(p.Value), StringComparer.Ordinal);
        using var archive = ZipFile.OpenRead(Path.Combine(modelDirectory, "weights.npz"));
        var sizes = Npy.ReadInt64(archive.GetEntry("sizes.npy") ?? throw new InvalidDataException("weights.npz has no sizes."));
        var weights = Npy.ReadFloat64(archive.GetEntry("w.npy") ?? throw new InvalidDataException("weights.npz has no w."));
        if (sizes[0] != tags.Count) throw new InvalidDataException("pkuseg's tag count disagrees with its features.");
        return new Pkuseg(unigram, features, tags, checked((int)sizes[1]), weights);
    }

    /// <summary>The <c>spacy_ontonotes</c> model's directory: pkuseg's own cache when it holds the model, else
    /// <paramref name="cacheDirectory"/> (default <c>~/.aidotnet/models/pkuseg</c>), downloading and checking it there
    /// first if needed.</summary>
    public static string DefaultModelDirectory(string? cacheDirectory = null)
    {
        static bool Complete(string directory) =>
            File.Exists(Path.Combine(directory, "features.msgpack")) && File.Exists(Path.Combine(directory, "weights.npz"));
        string pkusegHome = Environment.GetEnvironmentVariable("PKUSEG_HOME")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".pkuseg");
        string existing = Path.Combine(pkusegHome, DefaultModel);
        if (cacheDirectory is null && Complete(existing)) return existing;
        string target = Path.Combine(cacheDirectory ?? VerifiedDownload.CacheDirectory("pkuseg"), DefaultModel);
        VerifiedDownload.Zip(ModelUrl, target, ModelSha256, 256L << 20, "features.msgpack", "weights.npz");
        return target;
    }

    private static HashSet<string> LoadDictionaryWords()
    {
        // Postprocesser: pickle.load(default.pkl).strip().split("\n").
        var lines = ChineseText.ReadLines("pkuseg_default_dict.txt.gz").ToList();
        if (lines.Count > 0) lines[0] = lines[0].TrimStart();
        if (lines.Count > 0) lines[lines.Count - 1] = lines[lines.Count - 1].TrimEnd();
        return new HashSet<string>(lines, StringComparer.Ordinal);
    }

    /// <summary><c>pkuseg.cut(text)</c>: the text's words.</summary>
    public List<string> Cut(string text)
    {
        var words = new List<string>();
        var fragment = new StringBuilder();
        foreach (char c in text + " ")
        {
            if (!PythonString.IsSpace(c))
            {
                fragment.Append(c);
                continue;
            }
            if (fragment.Length == 0) continue;
            // No user dictionary: the preprocessor passes each fragment through whole.
            words.AddRange(PostProcess(CutFragment(fragment.ToString())));
            fragment.Clear();
        }
        return words;
    }

    // FeatureExtractor.normalize_text: keyword_rename, then the number and letter classes.
    private static string Normalize(string character)
    {
        if (character.Length == 1 && "-._,|/*:".IndexOf(character[0]) >= 0) character = "&";
        if (Numbers.Contains(character)) return "**Num";
        if (Letters.Contains(character)) return "**Letter";
        return character;
    }

    private List<string> CutFragment(string text)
    {
        var characters = ChineseText.Codepoints(text);
        var nodes = characters.Select(Normalize).ToList();
        int length = nodes.Count;
        var nodeScore = new double[length, _tags];
        for (int i = 0; i < length; i++)
            foreach (int feature in NodeFeatures(i, nodes))
                for (int s = 0; s < _tags; s++) nodeScore[i, s] += _weights[feature * _tags + s];
        // getLogYY: edge scores start at one.
        var edgeScore = new double[_tags, _tags];
        int backoff = _featureCount * _tags;
        for (int s = 0; s < _tags; s++)
            for (int pre = 0; pre < _tags; pre++) edgeScore[pre, s] = 1 + _weights[backoff + s * _tags + pre];
        var tags = Viterbi(nodeScore, edgeScore);

        var words = new List<string>();
        var current = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            if (i > 0 && _beginsWord[tags[i]])
            {
                words.Add(current.ToString());
                current.Clear();
            }
            current.Append(characters[i]);
        }
        if (current.Length > 0) words.Add(current.ToString());
        return words;
    }

    // run_viterbi: scored from the last node back; ties keep the later predecessor (>=), the first tag wins at the start.
    private int[] Viterbi(double[,] node, double[,] edge)
    {
        int w = node.GetLength(0), h = _tags;
        var max = new double[w, h];
        var previous = new int[w, h];
        for (int y = 0; y < h; y++) max[w - 1, y] = node[w - 1, y];
        for (int i = w - 2; i >= 0; i--)
            for (int y = 0; y < h; y++)
                for (int pre = 0; pre < h; pre++)
                {
                    double score = max[i + 1, pre] + node[i, y] + edge[y, pre];
                    if (pre == 0 || score >= max[i, y])
                    {
                        max[i, y] = score;
                        previous[i, y] = pre;
                    }
                }
        var states = new int[w];
        int tag = 0;
        for (int y = 1; y < h; y++)
            if (max[0, tag] < max[0, y]) tag = y;
        states[0] = tag;
        for (int i = 1; i < w; i++) states[i] = tag = previous[i - 1, tag];
        return states;
    }

    private static string Slice(List<string> nodes, int start, int count)
    {
        if (start < 0 || start >= nodes.Count || start + count >= nodes.Count + 1) return "";
        return string.Concat(nodes.Skip(start).Take(count));
    }

    // __get_node_features_idx.
    private List<int> NodeFeatures(int i, List<string> nodes)
    {
        var list = new List<int> { 0 };
        void Add(string feature)
        {
            if (_features.TryGetValue(feature, out int index)) list.Add(index);
        }
        int length = nodes.Count;
        string c = nodes[i];
        Add("c." + c);
        if (i > 0)
        {
            Add("c-1." + nodes[i - 1]);
            Add("c-1c." + nodes[i - 1] + "." + c);
        }
        if (i + 1 < length)
        {
            Add("c1." + nodes[i + 1]);
            Add("cc1." + c + "." + nodes[i + 1]);
        }
        if (i > 1)
        {
            Add("c-2." + nodes[i - 2]);
            Add("c-2c-1." + nodes[i - 2] + "." + nodes[i - 1]);
        }
        if (i + 2 < length) Add("c2." + nodes[i + 2]);

        var preIn = new List<string>();
        for (int l = WordMax; l >= WordMin; l--)
        {
            string piece = Slice(nodes, i - l + 1, l);
            if (_unigram.Contains(piece))
            {
                Add("w-1." + piece);
                preIn.Add(piece);
            }
            else preIn.Add(NoWord);
        }
        var postIn = new List<string>();
        for (int l = WordMax; l >= WordMin; l--)
        {
            string piece = Slice(nodes, i, l);
            if (_unigram.Contains(piece))
            {
                Add("w1." + piece);
                postIn.Add(piece);
            }
            else postIn.Add(NoWord);
        }
        var preEx = new List<string>();
        for (int l = WordMax; l >= WordMin; l--)
        {
            string piece = Slice(nodes, i - l, l);
            preEx.Add(_unigram.Contains(piece) ? piece : NoWord);
        }
        var postEx = new List<string>();
        for (int l = WordMax; l >= WordMin; l--)
        {
            string piece = Slice(nodes, i + 1, l);
            postEx.Add(_unigram.Contains(piece) ? piece : NoWord);
        }
        foreach (var pre in preEx)
            foreach (var post in postIn) Add("ww.l." + pre + "*" + post);
        foreach (var pre in preIn)
            foreach (var post in postEx) Add("ww.r." + pre + "*" + post);
        return list;
    }

    // Postprocesser.post_process(sent, check_seperated=True) with only the bundled word list.
    private static List<string> PostProcess(List<string> sentence)
    {
        var dictionary = DictionaryWords.Value;
        for (int m = 7; m >= 2; m--)
        {
            int end = sentence.Count - m;
            if (end < 0) continue;
            int i = 0;
            while (i < end + 1)
            {
                string merged = string.Concat(sentence.Skip(i).Take(m));
                bool merge = dictionary.Contains(merged) && !sentence.Skip(i).Take(m).All(dictionary.Contains);
                if (merge)
                {
                    sentence.RemoveRange(i, Math.Min(m, sentence.Count - i));
                    sentence.Insert(i, merged);
                    i++;
                    end = sentence.Count - m;
                }
                else
                {
                    i++;
                }
            }
        }
        return sentence;
    }

    /// <summary>NumPy <c>.npy</c> arrays (little-endian, C order) of <c>int64</c> and <c>float64</c>.</summary>
    private static class Npy
    {
        private static (string Descr, byte[] Data) Raw(ZipArchiveEntry entry)
        {
            using var stream = entry.Open();
            using var memory = new MemoryStream();
            stream.CopyTo(memory);
            var bytes = memory.ToArray();
            if (bytes.Length < 10 || bytes[0] != 0x93 || Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
                throw new InvalidDataException($"{entry.FullName} is not a NumPy array.");
            int major = bytes[6];
            int headerLength = major == 1 ? BitConverter.ToUInt16(bytes, 8) : BitConverter.ToInt32(bytes, 8);
            int start = major == 1 ? 10 : 12;
            string header = Encoding.ASCII.GetString(bytes, start, headerLength);
            if (header.Contains("'fortran_order': True")) throw new InvalidDataException($"{entry.FullName} is in Fortran order.");
            int at = header.IndexOf("'descr':", StringComparison.Ordinal);
            int open = header.IndexOf('\'', at + 8), close = header.IndexOf('\'', open + 1);
            string descr = header.Substring(open + 1, close - open - 1);
            var data = new byte[bytes.Length - start - headerLength];
            System.Array.Copy(bytes, start + headerLength, data, 0, data.Length);
            return (descr, data);
        }

        public static double[] ReadFloat64(ZipArchiveEntry entry)
        {
            var (descr, data) = Raw(entry);
            if (descr != "<f8") throw new InvalidDataException($"{entry.FullName} is {descr}, not float64.");
            var values = new double[data.Length / 8];
            Buffer.BlockCopy(data, 0, values, 0, values.Length * 8);
            return values;
        }

        public static long[] ReadInt64(ZipArchiveEntry entry)
        {
            var (descr, data) = Raw(entry);
            if (descr is "<i8") return Enumerable.Range(0, data.Length / 8).Select(i => BitConverter.ToInt64(data, i * 8)).ToArray();
            if (descr is "<i4") return Enumerable.Range(0, data.Length / 4).Select(i => (long)BitConverter.ToInt32(data, i * 4)).ToArray();
            throw new InvalidDataException($"{entry.FullName} is {descr}, not an integer array.");
        }
    }
}
