using System.IO.Compression;
using System.Text;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Open JTalk 1.11's rule tables (the <c>*_rule_utf_8.h</c> headers, as pyopenjtalk 0.3.4 vendors them), exported by
/// <c>tools/reference-data/openjtalk_rules.py</c>: each <c>#define</c> by its macro name and each string list with its
/// NULL entries, so the port's loops stop where the C loops do.
/// </summary>
internal static class OpenJTalkRules
{
    private static readonly Lazy<(Dictionary<string, string> Defines, Dictionary<string, string?[]> Lists)> Data = new(() =>
    {
        using var stream = typeof(OpenJTalkRules).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.openjtalk_rules.json.gz")
            ?? throw new InvalidDataException("The embedded Open JTalk rule tables are missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, Encoding.UTF8);
        var json = JObject.Parse(reader.ReadToEnd());
        var defines = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var p in ((JObject)json["defines"]!).Properties()) defines[p.Name] = (string)p.Value!;
        var lists = new Dictionary<string, string?[]>(StringComparer.Ordinal);
        foreach (var p in ((JObject)json["lists"]!).Properties())
            lists[p.Name] = ((JArray)p.Value).Select(v => v.Type == JTokenType.Null ? null : (string?)v).ToArray();
        return (defines, lists);
    }, isThreadSafe: true);

    /// <summary>A <c>#define</c>'s string.</summary>
    public static string D(string name) => Data.Value.Defines[name];

    /// <summary>A string list, NULLs included.</summary>
    public static string?[] L(string name) => Data.Value.Lists[name];

    /// <summary><c>strtopcmp</c>: the pattern's length when <paramref name="s"/> continues with it at
    /// <paramref name="at"/>, else -1 (an empty pattern matches with 0).</summary>
    public static int StrTopCmp(string s, int at, string pattern) =>
        at <= s.Length && string.CompareOrdinal(s, at, pattern, 0, pattern.Length) == 0 && s.Length - at >= pattern.Length ? pattern.Length : -1;

    /// <summary>C's <c>atoi</c>: leading whitespace, an optional sign, then digits; 0 when none.</summary>
    public static int Atoi(string? s)
    {
        if (s is null) return 0;
        int i = 0;
        while (i < s.Length && (s[i] == ' ' || (s[i] >= '\t' && s[i] <= '\r'))) i++;
        bool negative = false;
        if (i < s.Length && (s[i] == '+' || s[i] == '-')) negative = s[i++] == '-';
        long value = 0;
        while (i < s.Length && s[i] >= '0' && s[i] <= '9') value = unchecked(value * 10 + (s[i++] - '0'));
        return unchecked((int)(negative ? -value : value));
    }
}

/// <summary>
/// One word of pyopenjtalk's <c>run_frontend</c> result (<c>node2feature</c>): an NJD node's fields, "*" for each
/// unset string.
/// </summary>
internal sealed class NjdFeature
{
    public string String { get; set; } = "*";
    public string Pos { get; set; } = "*";
    public string PosGroup1 { get; set; } = "*";
    public string PosGroup2 { get; set; } = "*";
    public string PosGroup3 { get; set; } = "*";
    public string CType { get; set; } = "*";
    public string CForm { get; set; } = "*";
    public string Orig { get; set; } = "*";
    public string Read { get; set; } = "*";
    public string Pron { get; set; } = "*";
    public int Acc { get; set; }
    public int MoraSize { get; set; }
    public string ChainRule { get; set; } = "*";
    public int ChainFlag { get; set; } = -1;

    public NjdFeature Clone() => (NjdFeature)MemberwiseClone();
}

/// <summary>Open JTalk's <c>NJDNode</c> (njd/njd_node.c): a word's fields, a null string read back as "*".</summary>
internal sealed class NjdNode
{
    private const string NoData = "*";

    private string? _string, _pos, _posGroup1, _posGroup2, _posGroup3, _ctype, _cform, _orig, _read, _pron, _chainRule;
    private int _acc, _moraSize;

    public int ChainFlag { get; set; } = -1;
    public NjdNode? Prev { get; set; }
    public NjdNode? Next { get; set; }

    // NJDNode_set_*: a null or empty string unsets the field.
    private static string? Set(string? value) => string.IsNullOrEmpty(value) ? null : value;
    // NJDNode_add_*: appended (a null field takes the string as it is).
    private static string? Add(string? field, string? value) => value is null ? field : field is null ? value : field + value;

    public string String { get => _string ?? NoData; set => _string = Set(value); }
    public string Pos { get => _pos ?? NoData; set => _pos = Set(value); }
    public string PosGroup1 { get => _posGroup1 ?? NoData; set => _posGroup1 = Set(value); }
    public string PosGroup2 { get => _posGroup2 ?? NoData; set => _posGroup2 = Set(value); }
    public string PosGroup3 { get => _posGroup3 ?? NoData; set => _posGroup3 = Set(value); }
    public string CType { get => _ctype ?? NoData; set => _ctype = Set(value); }
    public string CForm { get => _cform ?? NoData; set => _cform = Set(value); }
    public string Orig { get => _orig ?? NoData; set => _orig = Set(value); }
    public string Read { get => _read ?? NoData; set => _read = Set(value); }
    public string Pron { get => _pron ?? NoData; set => _pron = Set(value); }
    public string ChainRule { get => _chainRule ?? NoData; set => _chainRule = Set(value); }

    // NJDNode_set_acc / set_mora_size: a negative value becomes 0.
    public int Acc { get => _acc; set => _acc = value < 0 ? 0 : value; }
    public int MoraSize { get => _moraSize; set => _moraSize = value < 0 ? 0 : value; }

    public void AddString(string? s) => _string = Add(_string, s);
    public void AddOrig(string? s) => _orig = Add(_orig, s);
    public void AddRead(string? s) => _read = Add(_read, s);
    public void AddPron(string? s) => _pron = Add(_pron, s);
    public void AddMoraSize(int size) => MoraSize = _moraSize + size;

    /// <summary>NJDNode_copy.</summary>
    public void CopyFrom(NjdNode other)
    {
        _string = other._string;
        _pos = other._pos;
        _posGroup1 = other._posGroup1;
        _posGroup2 = other._posGroup2;
        _posGroup3 = other._posGroup3;
        _ctype = other._ctype;
        _cform = other._cform;
        _orig = other._orig;
        _read = other._read;
        _pron = other._pron;
        Acc = other._acc;
        MoraSize = other._moraSize;
        _chainRule = other._chainRule;
        ChainFlag = other.ChainFlag;
    }

    // njd_node.c's get_token_from_string: up to the delimiter (consumed), no whitespace handling.
    private static string Token(string str, ref int index, char d)
    {
        int start = index;
        if (index < str.Length)
        {
            while (index < str.Length && str[index] != d) index++;
            int end = index;
            if (index < str.Length) index++;
            return str.Substring(start, end - start);
        }
        return "";
    }

    /// <summary>
    /// NJDNode_load: a MeCab feature line (<c>surface,pos,...,acc/mora,chain_rule,chain_flag</c>) into this node; a
    /// chained word (several "/" in its accent field) becomes a run of nodes linked after it.
    /// </summary>
    public void Load(string str)
    {
        int index = 0;
        string buffString = Token(str, ref index, ',');
        Pos = Token(str, ref index, ',');
        PosGroup1 = Token(str, ref index, ',');
        PosGroup2 = Token(str, ref index, ',');
        PosGroup3 = Token(str, ref index, ',');
        CType = Token(str, ref index, ',');
        CForm = Token(str, ref index, ',');
        string buffOrig = Token(str, ref index, ',');
        string buffRead = Token(str, ref index, ',');
        string buffPron = Token(str, ref index, ',');
        string buffAcc = Token(str, ref index, ',');
        ChainRule = Token(str, ref index, ',');
        string flag = Token(str, ref index, ',');
        if (flag == "1") ChainFlag = 1;
        else if (flag == "0") ChainFlag = 0;

        // for symbol
        if (buffAcc.Contains("*") || !buffAcc.Contains("/"))
        {
            String = buffString;
            Orig = buffOrig;
            Read = buffRead;
            Pron = buffPron;
            Acc = 0;
            MoraSize = 0;
            return;
        }

        int count = buffAcc.Count(c => c == '/');
        if (count == 1)
        {
            String = buffString;
            Orig = buffOrig;
            Read = buffRead;
            Pron = buffPron;
            int indexAcc = 0;
            string acc = Token(buffAcc, ref indexAcc, '/');
            Acc = acc.Length == 0 ? 0 : OpenJTalkRules.Atoi(acc);
            string mora = Token(buffAcc, ref indexAcc, ':');
            MoraSize = mora.Length == 0 ? 0 : OpenJTalkRules.Atoi(mora);
            return;
        }

        // parse chained word; the string is cut by the byte lengths of the orig pieces, as the C code cuts it
        var stringBytes = Encoding.UTF8.GetBytes(buffString);
        int indexString = 0, indexOrig = 0, indexRead = 0, indexPron = 0, indexAccent = 0;
        NjdNode node = this;
        NjdNode? prev = null;
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
            {
                node = new NjdNode();
                node.CopyFrom(prev!);
                node.ChainFlag = 0;
                node.Prev = prev;
                prev!.Next = node;
            }
            string buff = Token(buffOrig, ref indexOrig, ':');
            node.Orig = buff;
            if (i + 1 < count)
            {
                node.String = buff;
                indexString += Encoding.UTF8.GetByteCount(buff);
            }
            else
            {
                int from = Math.Min(indexString, stringBytes.Length);
                node.String = Encoding.UTF8.GetString(stringBytes, from, stringBytes.Length - from);
            }
            node.Read = Token(buffRead, ref indexRead, ':');
            node.Pron = Token(buffPron, ref indexPron, ':');
            string acc = Token(buffAcc, ref indexAccent, '/');
            node.Acc = acc.Length == 0 ? 0 : OpenJTalkRules.Atoi(acc);
            string mora = Token(buffAcc, ref indexAccent, ':');
            node.MoraSize = mora.Length == 0 ? 0 : OpenJTalkRules.Atoi(mora);
            prev = node;
        }
    }

    /// <summary>NJDNode_insert: the run starting at <paramref name="node"/> between two nodes; its tail.</summary>
    public static NjdNode Insert(NjdNode? prev, NjdNode? next, NjdNode node)
    {
        if (prev is null || next is null)
            throw new InvalidOperationException("NJDNode_insert() in njd_node.c: NJDNodes are not specified.");
        var tail = node;
        while (tail.Next is not null) tail = tail.Next;
        prev.Next = node;
        node.Prev = prev;
        next.Prev = tail;
        tail.Next = next;
        return tail;
    }

    public NjdFeature ToFeature() => new()
    {
        String = String, Pos = Pos, PosGroup1 = PosGroup1, PosGroup2 = PosGroup2, PosGroup3 = PosGroup3, CType = CType,
        CForm = CForm, Orig = Orig, Read = Read, Pron = Pron, Acc = Acc, MoraSize = MoraSize, ChainRule = ChainRule,
        ChainFlag = ChainFlag,
    };

    public static NjdNode FromFeature(NjdFeature f) => new()
    {
        String = f.String, Pos = f.Pos, PosGroup1 = f.PosGroup1, PosGroup2 = f.PosGroup2, PosGroup3 = f.PosGroup3,
        CType = f.CType, CForm = f.CForm, Orig = f.Orig, Read = f.Read, Pron = f.Pron, Acc = f.Acc, MoraSize = f.MoraSize,
        ChainRule = f.ChainRule, ChainFlag = f.ChainFlag,
    };
}

/// <summary>Open JTalk's <c>NJD</c> (njd/njd.c): the doubly linked list of words.</summary>
internal sealed class Njd
{
    public NjdNode? Head { get; private set; }
    public NjdNode? Tail { get; private set; }

    /// <summary>NJD_push_node: a node (and any nodes already linked after it) at the end.</summary>
    public void Push(NjdNode node)
    {
        if (Head is null) Head = node;
        else
        {
            Tail!.Next = node;
            node.Prev = Tail;
        }
        while (node.Next is not null) node = node.Next;
        Tail = node;
    }

    /// <summary>NJD_remove_node: removes a node; the next one.</summary>
    public NjdNode? Remove(NjdNode node)
    {
        NjdNode? next;
        if (node == Head && node == Tail)
        {
            Head = null;
            Tail = null;
            next = null;
        }
        else if (node == Head)
        {
            Head = node.Next;
            Head!.Prev = null;
            next = Head;
        }
        else if (node == Tail)
        {
            Tail = node.Prev;
            Tail!.Next = null;
            next = null;
        }
        else
        {
            node.Prev!.Next = node.Next;
            node.Next!.Prev = node.Prev;
            next = node.Next;
        }
        node.Prev = null;
        node.Next = null;
        return next;
    }

    /// <summary>NJD_remove_silent_node: every node whose pronunciation is unset.</summary>
    public void RemoveSilentNodes()
    {
        for (var node = Head; node is not null;)
            node = node.Pron == "*" ? Remove(node) : node.Next;
    }

    public List<NjdFeature> ToFeatures()
    {
        var features = new List<NjdFeature>();
        for (var node = Head; node is not null; node = node.Next) features.Add(node.ToFeature());
        return features;
    }
}
