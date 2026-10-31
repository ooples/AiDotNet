using System.Linq;
using System.IO.Compression;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// MeCab 0.996's morphological segmentation (<c>MeCab.Tagger("-Owakati")</c>) with the unidic-lite 1.0.8 dictionary:
/// a lattice of the dictionary's words and the unknown-word candidates of each character category over the text's UTF-8
/// bytes, and the Viterbi path by connection and word costs. <see cref="KoDic"/> is mecab-ko 0.996-ko-0.9.2 with
/// mecab-ko-dic 2.1.1 (python-mecab-ko's <c>MeCab().pos</c>): the same lattice, mecab-ko's left-space penalty added to
/// the connection cost, and each word's part-of-speech tag.
/// </summary>
/// <remarks>
/// <para>The dictionary's segmentation data (surfaces, context ids, word costs, the connection matrix, character
/// categories and unknown-word entries; not its features) is exported from unidic-lite's binary files by
/// <c>tools/reference-data/mecab_unidic_reference.py</c>. The lattice follows MeCab's <c>Tokenizer::lookup</c> and
/// <c>Viterbi::viterbi</c> exactly, including the order nodes are chained in (which decides ties) and its reading of the
/// buffer's terminating NUL at the end of the text.</para>
/// </remarks>
internal sealed class MeCab
{
    private static readonly Lazy<MeCab> SharedInstance = new(() => new MeCab("mecab_unidic_lite.gz"), isThreadSafe: true);
    private static readonly Lazy<MeCab> KoDicInstance = new(() => new MeCab("mecab_ko_dic.gz"), isThreadSafe: true);

    /// <summary>The shared instance (the dictionary loads once, on first use).</summary>
    public static MeCab UnidicLite => SharedInstance.Value;

    /// <summary>mecab-ko with mecab-ko-dic (exported by <c>tools/reference-data/mecab_ko_reference.py</c>).</summary>
    public static MeCab KoDic => KoDicInstance.Value;

    private const int MaxGroupingSize = 24;   // DEFAULT_MAX_GROUPING_SIZE; unidic-lite's dicrc leaves it unset

    private sealed class Lexicon
    {
        public readonly Dictionary<string, int> Keys = new(StringComparer.Ordinal);   // key bytes as Latin-1 characters
        public int MaxKeyLength;
        public ushort[] Left = Array.Empty<ushort>();
        public ushort[] Right = Array.Empty<ushort>();
        public short[] Cost = Array.Empty<short>();
        public ushort[] PosId = Array.Empty<ushort>();
        public ushort[] Tag = Array.Empty<ushort>();
        // A dictionary read from MeCab's binary format: the Darts double array its keys are searched in, and each
        // token's feature (an offset into the NUL-terminated feature strings).
        public int[]? Base;
        public uint[]? Check;
        public uint[] Feature = Array.Empty<uint>();
        public byte[] Features = Array.Empty<byte>();

        public string FeatureOf(int token)
        {
            int start = (int)Feature[token], end = start;
            while (Features[end] != 0) end++;
            return Encoding.UTF8.GetString(Features, start, end - start);
        }

        // Darts::DoubleArray::commonPrefixSearch: each key the len bytes from begin start with, shortest first.
        public void CommonPrefixSearch(byte[] s, int begin, int len, List<(int Value, int Length)> results)
        {
            if (Base is null || Check is null)
            {
                var key = new StringBuilder(len);
                for (int length = 1; length <= Math.Min(len, MaxKeyLength); length++)
                {
                    key.Append((char)s[begin + length - 1]);
                    if (Keys.TryGetValue(key.ToString(), out int value)) results.Add((value, length));
                }
                return;
            }
            int b = Base[0];
            for (int i = 0; i < len; i++)
            {
                if ((uint)b == Check[b] && Base[b] < 0) results.Add((-Base[b] - 1, i));
                int p = b + s[begin + i] + 1;
                if (p >= Check.Length || (uint)b != Check[p]) return;
                b = Base[p];
            }
            if ((uint)b == Check[b] && Base[b] < 0) results.Add((-Base[b] - 1, len));
        }

        // Darts::DoubleArray::exactMatchSearch; -1 when the key is absent.
        public int ExactMatchSearch(byte[] key)
        {
            if (Base is null || Check is null) return Keys.TryGetValue(Latin1(key), out int found) ? found : -1;
            int b = Base[0];
            foreach (byte c in key)
            {
                int p = b + c + 1;
                if (p >= Check.Length || (uint)b != Check[p]) return -1;
                b = Base[p];
            }
            return (uint)b == Check[b] && Base[b] < 0 ? -Base[b] - 1 : -1;
        }
    }

    private readonly Lexicon _system;
    private readonly List<Lexicon> _user = new();
    private readonly Lexicon _unknown;
    private readonly (int Token, int Count)[] _unknownTokens;
    private readonly int _leftSize;
    private readonly short[] _matrix;
    private readonly uint[] _charMap;
    private readonly uint _space;
    private readonly string[] _tags = Array.Empty<string>();
    private readonly Dictionary<int, int> _leftSpacePenalty = new();

    private MeCab(string resource)
    {
        using var stream = typeof(MeCab).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd." + resource)
            ?? throw new InvalidDataException($"The embedded resource {resource} is missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new BinaryReader(new BufferedStream(gzip, 1 << 20));
        string format = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (format != "MCB1" && format != "MCB2") throw new InvalidDataException("Unknown MeCab data format.");
        bool tagged = format == "MCB2";
        _system = ReadLexicon(reader, tagged);
        _unknown = ReadLexicon(reader, tagged);
        int categories = reader.ReadInt32();
        var names = new string[categories];
        for (int i = 0; i < categories; i++) names[i] = Latin1(reader.ReadBytes(reader.ReadByte()));
        _unknownTokens = new (int, int)[categories];
        for (int i = 0; i < categories; i++)
        {
            if (!_unknown.Keys.TryGetValue(names[i], out int value))
                throw new InvalidDataException($"cannot find UNK category: {names[i]}");
            _unknownTokens[i] = (value >> 8, value & 0xff);
        }
        _leftSize = reader.ReadUInt16();
        int rightSize = reader.ReadUInt16();
        _matrix = new short[_leftSize * rightSize];
        var row = reader.ReadBytes(_matrix.Length * 2);
        // Differences down each column, accumulated in 16-bit arithmetic as they were taken.
        for (int r = 0; r < rightSize; r++)
            for (int c = 0; c < _leftSize; c++)
            {
                int i = r * _leftSize + c;
                short delta = (short)(row[2 * i] | (row[2 * i + 1] << 8));
                _matrix[i] = r == 0 ? delta : unchecked((short)(_matrix[i - _leftSize] + delta));
            }
        _charMap = new uint[0xffff];
        for (int i = 0; i < _charMap.Length; i++) _charMap[i] = reader.ReadUInt32();
        _space = _charMap[0x20];
        if (!tagged) return;
        _tags = new string[reader.ReadUInt16()];
        for (int i = 0; i < _tags.Length; i++) _tags[i] = Encoding.UTF8.GetString(reader.ReadBytes(reader.ReadByte()));
        // left-space-penalty-factor: the first listed cost for a posid applies.
        int penalties = reader.ReadUInt16();
        for (int i = 0; i < penalties; i++)
        {
            int posid = reader.ReadUInt16(), cost = reader.ReadInt32();
            if (!_leftSpacePenalty.ContainsKey(posid)) _leftSpacePenalty[posid] = cost;
        }
    }

    /// <summary>
    /// A MeCab 0.996 dictionary directory as <c>mecab-dict-index</c> writes it (<c>sys.dic</c>, <c>unk.dic</c>,
    /// <c>matrix.bin</c>, <c>char.bin</c>), read as MeCab's <c>Dictionary::open</c>, <c>Connector::open</c> and
    /// <c>CharProperty::open</c> read them; Open JTalk's (<c>open_jtalk_dic_utf_8-1.11</c>) for one. Its words come
    /// with their features (<see cref="ParseFeatures"/>).
    /// </summary>
    public static MeCab Open(string directory)
    {
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        return new MeCab(directory, fromDirectory: true);
    }

    /// <summary>
    /// <see cref="Open"/> with a user dictionary (<c>mecab -u</c>), compiled from its CSV source as
    /// <c>mecab-dict-index -u</c> compiles it: each line's surface, left and right context ids, cost and feature (MeCab's
    /// <c>tokenizeCSV</c> into five fields, the feature the rest of the line), the entries stably sorted by surface bytes.
    /// Its words join the system dictionary's at each position, after them.
    /// </summary>
    public static MeCab Open(string directory, IEnumerable<string> userDictionaryCsv)
    {
        if (userDictionaryCsv is null) throw new ArgumentNullException(nameof(userDictionaryCsv));
        var mecab = Open(directory);
        mecab._user.Add(mecab.CompileUserDictionary(userDictionaryCsv));
        return mecab;
    }

    // MeCab's tokenizeCSV(line, col, max): fields split at commas after skipping leading blanks, a quoted field unquoted
    // ("" for a quote); the last field is the rest of the line as written.
    private static List<string> TokenizeCsv(string line, int max)
    {
        var fields = new List<string>();
        int at = 0;
        while (at < line.Length)
        {
            while (at < line.Length && (line[at] == ' ' || line[at] == '\t')) at++;
            if (max == 1)
            {
                fields.Add(line.Substring(at));
                break;
            }
            string field;
            if (at < line.Length && line[at] == '"')
            {
                var value = new StringBuilder();
                at++;
                for (; at < line.Length; at++)
                {
                    if (line[at] == '"')
                    {
                        at++;
                        if (at >= line.Length || line[at] != '"') break;
                    }
                    value.Append(line[at]);
                }
                int comma = line.IndexOf(',', Math.Min(at, line.Length));
                at = comma < 0 ? line.Length : comma;
                field = value.ToString();
            }
            else
            {
                int comma = line.IndexOf(',', at);
                int end = comma < 0 ? line.Length : comma;
                field = line.Substring(at, end - at);
                at = end;
            }
            fields.Add(field);
            max--;
            at++;
        }
        return fields;
    }

    // mecab's toInt: atoi, or INT_MAX for an empty field.
    private static int ToInt(string field)
    {
        if (field.Length == 0) return int.MaxValue;
        int at = 0;
        while (at < field.Length && char.IsWhiteSpace(field[at])) at++;
        bool negative = false;
        if (at < field.Length && (field[at] == '+' || field[at] == '-')) negative = field[at++] == '-';
        long value = 0;
        while (at < field.Length && field[at] >= '0' && field[at] <= '9' && value <= int.MaxValue) value = value * 10 + (field[at++] - '0');
        return (int)(negative ? -value : value);
    }

    private Lexicon CompileUserDictionary(IEnumerable<string> lines)
    {
        var entries = new List<(string Key, ushort Left, ushort Right, short Cost, string Feature)>();
        int rightSize = _matrix.Length / _leftSize;
        foreach (var line in lines)
        {
            var col = TokenizeCsv(line, 5);
            if (col.Count != 5) throw new InvalidDataException($"format error: {line}");
            int lid = ToInt(col[1]), rid = ToInt(col[2]), cost = ToInt(col[3]);
            // An empty cost or context id needs the dictionary's rewrite rules and feature model, which a compiled
            // dictionary directory does not carry; every entry GPT-SoVITS's user dictionary writes has them.
            if (cost == int.MaxValue || lid == int.MaxValue || rid == int.MaxValue || lid < 0 || rid < 0)
                throw new InvalidDataException($"A user dictionary entry without its cost or context ids is not supported: {line}");
            if (lid >= _leftSize || rid >= rightSize) throw new InvalidDataException($"invalid ids are found lid={lid} rid={rid}");
            if (col[0].Length == 0) continue;
            entries.Add((Latin1(Encoding.UTF8.GetBytes(col[0])), (ushort)lid, (ushort)rid, (short)cost, col[4]));
        }
        var sorted = entries.OrderBy(e => e.Key, StringComparer.Ordinal).ToList();
        var lexicon = new Lexicon
        {
            Left = sorted.Select(e => e.Left).ToArray(),
            Right = sorted.Select(e => e.Right).ToArray(),
            Cost = sorted.Select(e => e.Cost).ToArray(),
            Feature = new uint[sorted.Count],
        };
        var features = new List<byte>();
        for (int i = 0; i < sorted.Count; i++)
        {
            lexicon.Feature[i] = (uint)features.Count;
            features.AddRange(Encoding.UTF8.GetBytes(sorted[i].Feature));
            features.Add(0);
        }
        lexicon.Features = features.ToArray();
        for (int i = 0; i < sorted.Count;)
        {
            int j = i;
            while (j < sorted.Count && sorted[j].Key == sorted[i].Key) j++;
            lexicon.Keys[sorted[i].Key] = (i << 8) + (j - i);
            lexicon.MaxKeyLength = Math.Max(lexicon.MaxKeyLength, sorted[i].Key.Length);
            i = j;
        }
        return lexicon;
    }

    private MeCab(string directory, bool fromDirectory)
    {
        _system = ReadDictionary(Path.Combine(directory, "sys.dic"), out int leftSize, out int rightSize);
        _unknown = ReadDictionary(Path.Combine(directory, "unk.dic"), out _, out _);

        var matrix = File.ReadAllBytes(Path.Combine(directory, "matrix.bin"));
        _leftSize = BitConverter.ToUInt16(matrix, 0);
        int matrixRight = BitConverter.ToUInt16(matrix, 2);
        if (_leftSize != leftSize || matrixRight != rightSize || matrix.Length != 2 * (_leftSize * matrixRight + 2))
            throw new InvalidDataException("matrix.bin does not match sys.dic.");
        _matrix = new short[_leftSize * matrixRight];
        Buffer.BlockCopy(matrix, 4, _matrix, 0, _matrix.Length * 2);

        var chars = File.ReadAllBytes(Path.Combine(directory, "char.bin"));
        int categories = (int)BitConverter.ToUInt32(chars, 0);
        if (chars.Length != 4 + 32 * categories + 4 * 0xffff) throw new InvalidDataException("char.bin: invalid file size.");
        _unknownTokens = new (int, int)[categories];
        for (int i = 0; i < categories; i++)
        {
            int start = 4 + 32 * i, end = start;
            while (end < start + 32 && chars[end] != 0) end++;
            var name = new byte[end - start];
            Array.Copy(chars, start, name, 0, name.Length);
            int value = _unknown.ExactMatchSearch(name);
            if (value == -1) throw new InvalidDataException($"cannot find UNK category: {Encoding.UTF8.GetString(name)}");
            _unknownTokens[i] = (value >> 8, value & 0xff);
        }
        _charMap = new uint[0xffff];
        Buffer.BlockCopy(chars, 4 + 32 * categories, _charMap, 0, _charMap.Length * 4);
        _space = _charMap[0x20];
    }

    private const uint DictionaryMagicId = 0xef718f77;

    private static Lexicon ReadDictionary(string path, out int leftSize, out int rightSize)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 100) throw new InvalidDataException($"dictionary file is broken: {path}");
        uint Header(int i) => BitConverter.ToUInt32(bytes, 4 * i);
        if ((Header(0) ^ DictionaryMagicId) != (uint)bytes.Length) throw new InvalidDataException($"dictionary file is broken: {path}");
        if (Header(1) != 102) throw new InvalidDataException($"incompatible version: {Header(1)}");
        leftSize = (int)Header(4);
        rightSize = (int)Header(5);
        int dsize = (int)Header(6), tsize = (int)Header(7), fsize = (int)Header(8);
        int offset = 40 + 32;
        if (offset + dsize + tsize + fsize != bytes.Length) throw new InvalidDataException($"dictionary file is broken: {path}");

        var lexicon = new Lexicon { Base = new int[dsize / 8], Check = new uint[dsize / 8] };
        for (int i = 0; i < lexicon.Base.Length; i++)
        {
            lexicon.Base[i] = BitConverter.ToInt32(bytes, offset + 8 * i);
            lexicon.Check[i] = BitConverter.ToUInt32(bytes, offset + 8 * i + 4);
        }
        offset += dsize;
        int tokens = tsize / 16;
        lexicon.Left = new ushort[tokens];
        lexicon.Right = new ushort[tokens];
        lexicon.PosId = new ushort[tokens];
        lexicon.Cost = new short[tokens];
        lexicon.Feature = new uint[tokens];
        for (int i = 0; i < tokens; i++, offset += 16)
        {
            lexicon.Left[i] = BitConverter.ToUInt16(bytes, offset);
            lexicon.Right[i] = BitConverter.ToUInt16(bytes, offset + 2);
            lexicon.PosId[i] = BitConverter.ToUInt16(bytes, offset + 4);
            lexicon.Cost[i] = BitConverter.ToInt16(bytes, offset + 6);
            lexicon.Feature[i] = BitConverter.ToUInt32(bytes, offset + 8);
        }
        lexicon.Features = new byte[fsize];
        Array.Copy(bytes, offset, lexicon.Features, 0, fsize);
        return lexicon;
    }

    private static string Latin1(byte[] bytes)
    {
        var chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) chars[i] = (char)bytes[i];
        return new string(chars);
    }

    private static Lexicon ReadLexicon(BinaryReader reader, bool tagged)
    {
        var lexicon = new Lexicon();
        int count = reader.ReadInt32();
        var previous = Array.Empty<byte>();
        for (int i = 0; i < count; i++)
        {
            int shared = reader.ReadByte(), length = reader.ReadByte();
            var key = new byte[shared + length];
            Array.Copy(previous, key, shared);
            reader.Read(key, shared, length);
            lexicon.Keys[Latin1(key)] = reader.ReadInt32();
            lexicon.MaxKeyLength = Math.Max(lexicon.MaxKeyLength, key.Length);
            previous = key;
        }
        int tokens = reader.ReadInt32();
        lexicon.Left = new ushort[tokens];
        lexicon.Right = new ushort[tokens];
        lexicon.Cost = new short[tokens];
        for (int i = 0; i < tokens; i++) lexicon.Left[i] = reader.ReadUInt16();
        for (int i = 0; i < tokens; i++) lexicon.Right[i] = reader.ReadUInt16();
        for (int i = 0; i < tokens; i++) lexicon.Cost[i] = reader.ReadInt16();
        if (!tagged) return lexicon;
        lexicon.PosId = new ushort[tokens];
        lexicon.Tag = new ushort[tokens];
        for (int i = 0; i < tokens; i++) lexicon.PosId[i] = reader.ReadUInt16();
        for (int i = 0; i < tokens; i++) lexicon.Tag[i] = reader.ReadUInt16();
        return lexicon;
    }

    // ---------------------------------------------------------------- CharInfo (bit fields of a uint)

    private static uint Type(uint info) => info & 0x3ffff;
    private static int DefaultType(uint info) => (int)((info >> 18) & 0xff);
    private static int Length(uint info) => (int)((info >> 26) & 0xf);
    private static bool Group(uint info) => ((info >> 30) & 1) != 0;
    private static bool Invoke(uint info) => ((info >> 31) & 1) != 0;
    private static bool IsKindOf(uint a, uint b) => (Type(a) & Type(b)) != 0;

    // getCharInfo with utf8_to_ucs2: characters outside the BMP read as 0; at the end the terminating NUL is read.
    private uint CharInfo(byte[] s, int p, int end, out int mblen)
    {
        int len = end - p;
        int b0 = p < s.Length ? s[p] : 0;
        int code;
        if (b0 < 0x80) { mblen = 1; code = b0; }
        else if (len >= 2 && (b0 & 0xe0) == 0xc0) { mblen = 2; code = ((b0 & 0x1f) << 6) | (s[p + 1] & 0x3f); }
        else if (len >= 3 && (b0 & 0xf0) == 0xe0) { mblen = 3; code = ((b0 & 0x0f) << 12) | ((s[p + 1] & 0x3f) << 6) | (s[p + 2] & 0x3f); }
        else if (len >= 4 && (b0 & 0xf8) == 0xf0) { mblen = 4; code = 0; }
        else if (len >= 5 && (b0 & 0xfc) == 0xf8) { mblen = 5; code = 0; }
        else if (len >= 6 && (b0 & 0xfe) == 0xfc) { mblen = 6; code = 0; }
        else { mblen = 1; code = 0; }
        // MeCab's map holds 0xffff entries; U+FFFF reads past it (here: as NUL's category).
        return code < _charMap.Length ? _charMap[code] : _charMap[0];
    }

    // seekToOtherType: past the characters of c's kind; `fail` is the last character read.
    private int SeekToOtherType(byte[] s, int begin, int end, uint c, ref uint fail, ref int mblen, out int clen)
    {
        int p = begin;
        clen = 0;
        while (p != end && IsKindOf(c, fail = CharInfo(s, p, end, out mblen)))
        {
            p += mblen;
            clen++;
            c = fail;
        }
        return p;
    }

    // ---------------------------------------------------------------- the lattice

    private sealed class Node
    {
        public int Surface, Length, RLength;
        public int Left, Right, WordCost;
        public int PosId, Tag = -1;
        public Lexicon? Source;
        public int Token;
        public long Cost;
        public Node? Previous, BNext, ENext;
    }

    private Node? Lookup(byte[] s, int begin, int end)
    {
        if (end - begin >= 65535) end = begin + 65535;
        Node? result = null;
        uint cinfo = 0;
        int mblen = 0;
        int begin2 = SeekToOtherType(s, begin, end, _space, ref cinfo, ref mblen, out _);

        // commonPrefixSearch: every dictionary key the bytes from begin2 start with, shortest first (a zero length
        // reads up to the buffer's NUL, which is at the end of the text).
        // Tokenizer::lookup: the system dictionary's words, then each user dictionary's.
        foreach (var dictionary in new[] { _system }.Concat(_user))
        {
            var matches = new List<(int Value, int Length)>();
            dictionary.CommonPrefixSearch(s, begin2, end - begin2, matches);
            foreach (var (value, length) in matches)
            {
                int token = value >> 8, count = value & 0xff;
                for (int j = 0; j < count; j++)
                    result = new Node
                    {
                        Surface = begin2, Length = length, RLength = begin2 - begin + length, Left = dictionary.Left[token + j],
                        Right = dictionary.Right[token + j], WordCost = dictionary.Cost[token + j], BNext = result,
                        PosId = dictionary.PosId.Length > 0 ? dictionary.PosId[token + j] : 0,
                        Tag = dictionary.Tag.Length > 0 ? dictionary.Tag[token + j] : -1,
                        Source = dictionary, Token = token + j,
                    };
            }
        }
        if (result is not null && !Invoke(cinfo)) return result;

        void AddUnknown(int begin3)
        {
            var (token, count) = _unknownTokens[DefaultType(cinfo)];
            for (int k = 0; k < count; k++)
                result = new Node
                {
                    Surface = begin2, Length = begin3 - begin2, RLength = begin3 - begin, Left = _unknown.Left[token + k],
                    Right = _unknown.Right[token + k], WordCost = _unknown.Cost[token + k], BNext = result,
                    PosId = _unknown.PosId.Length > 0 ? _unknown.PosId[token + k] : 0,
                    Tag = _unknown.Tag.Length > 0 ? _unknown.Tag[token + k] : -1,
                    Source = _unknown, Token = token + k,
                };
        }

        int begin3 = begin2 + mblen;
        int groupBegin3 = -1;
        if (begin3 > end)
        {
            AddUnknown(begin3);
            if (result is not null) return result;
        }
        if (Group(cinfo))
        {
            int tmp = begin3;
            uint fail = 0;
            begin3 = SeekToOtherType(s, begin3, end, cinfo, ref fail, ref mblen, out int clen);
            if (clen <= MaxGroupingSize) AddUnknown(begin3);
            groupBegin3 = begin3;
            begin3 = tmp;
        }
        for (int i = 1; i <= Length(cinfo); i++)
        {
            if (begin3 > end) break;
            if (begin3 == groupBegin3) continue;
            AddUnknown(begin3);
            if (!IsKindOf(cinfo, CharInfo(s, begin3, end, out mblen))) break;
            begin3 += mblen;
        }
        if (result is null) AddUnknown(begin3);
        return result;
    }

    private void Connect(int position, Node? right, Node?[] endNodes)
    {
        for (; right is not null; right = right.BNext)
        {
            long best = int.MaxValue;
            Node? bestNode = null;
            // mecab-ko's Connector::get_space_penalty_cost: a node with leading space pays its posid's penalty.
            int penalty = right.RLength != right.Length && _leftSpacePenalty.TryGetValue(right.PosId, out int p) ? p : 0;
            for (var left = endNodes[position]; left is not null; left = left.ENext)
            {
                long cost = left.Cost + _matrix[left.Right + _leftSize * right.Left] + right.WordCost + penalty;
                if (cost < best)
                {
                    best = cost;
                    bestNode = left;
                }
            }
            if (bestNode is null) throw new InvalidOperationException("too long sentence.");
            right.Previous = bestNode;
            right.Cost = best;
            int x = right.RLength + position;
            right.ENext = endNodes[x];
            endNodes[x] = right;
        }
    }

    /// <summary>The best path's surfaces, in order.</summary>
    public IReadOnlyList<string> Segment(string text) => Parse(text).Select(w => w.Surface).ToList();

    /// <summary>The best path's words and their part-of-speech tags (empty for a dictionary without tags).</summary>
    public IReadOnlyList<(string Surface, string Tag)> Parse(string text) =>
        BestPath(text).Select(w => (w.Surface, w.Node.Tag >= 0 ? _tags[w.Node.Tag] : "")).ToList();

    /// <summary>
    /// The best path's words and their features (a dictionary read by <see cref="Open"/>): what Open JTalk's
    /// <c>Mecab_analysis</c> collects from the lattice, each word's surface and its feature string.
    /// </summary>
    public IReadOnlyList<(string Surface, string Feature)> ParseFeatures(string text) =>
        BestPath(text).Select(w => (w.Surface, w.Node.Source is { Features.Length: > 0 } source ? source.FeatureOf(w.Node.Token)
            : throw new InvalidOperationException("This dictionary has no features; open a MeCab dictionary directory."))).ToList();

    private List<(string Surface, Node Node)> BestPath(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        int nul = text.IndexOf('\0');
        if (nul >= 0) text = text.Substring(0, nul);   // the C API reads a NUL-terminated string
        var s = Encoding.UTF8.GetBytes(text);
        int len = s.Length;
        // A node can end past the text (an unknown word read beyond its last byte); room for those.
        var endNodes = new Node?[len + 8];
        endNodes[0] = new Node();
        for (int position = 0; position < len; position++)
            if (endNodes[position] is not null) Connect(position, Lookup(s, position, len), endNodes);
        var eos = new Node { Surface = len };
        for (int position = len; position >= 0; position--)
            if (endNodes[position] is not null)
            {
                Connect(position, eos, endNodes);
                break;
            }
        var words = new List<(string, Node)>();
        for (var node = eos.Previous; node is not null && node.Previous is not null; node = node.Previous)
            words.Add((Encoding.UTF8.GetString(s, node.Surface, Math.Min(node.Length, len - node.Surface)), node));
        words.Reverse();
        return words;
    }

    /// <summary><c>Tagger("-Owakati").parse(text)</c>: each surface followed by a space, then a newline.</summary>
    public string Wakati(string text)
    {
        var output = new StringBuilder();
        foreach (var word in Segment(text)) output.Append(word).Append(' ');
        return output.Append('\n').ToString();
    }
}
