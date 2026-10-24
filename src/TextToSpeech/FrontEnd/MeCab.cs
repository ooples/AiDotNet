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
    }

    private readonly Lexicon _system;
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
        int available = end - begin2;
        int limit = Math.Min(available, _system.MaxKeyLength);
        var key = new StringBuilder(limit);
        for (int length = 1; length <= limit; length++)
        {
            key.Append((char)s[begin2 + length - 1]);
            if (!_system.Keys.TryGetValue(key.ToString(), out int value)) continue;
            int token = value >> 8, count = value & 0xff;
            for (int j = 0; j < count; j++)
                result = new Node
                {
                    Surface = begin2, Length = length, RLength = begin2 - begin + length, Left = _system.Left[token + j],
                    Right = _system.Right[token + j], WordCost = _system.Cost[token + j], BNext = result,
                    PosId = _system.PosId.Length > 0 ? _system.PosId[token + j] : 0,
                    Tag = _system.Tag.Length > 0 ? _system.Tag[token + j] : -1,
                };
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
    public IReadOnlyList<(string Surface, string Tag)> Parse(string text)
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
        var words = new List<(string, string)>();
        for (var node = eos.Previous; node is not null && node.Previous is not null; node = node.Previous)
            words.Add((Encoding.UTF8.GetString(s, node.Surface, Math.Min(node.Length, len - node.Surface)),
                node.Tag >= 0 ? _tags[node.Tag] : ""));
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
