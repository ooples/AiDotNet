using System.Globalization;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Sudachi's character categories (<c>CategoryType</c>), as bit flags.</summary>
[Flags]
internal enum SudachiCategory : uint
{
    None = 0,
    Default = 1 << 0,
    Space = 1 << 1,
    Kanji = 1 << 2,
    Symbol = 1 << 3,
    Numeric = 1 << 4,
    Alpha = 1 << 5,
    Hiragana = 1 << 6,
    Katakana = 1 << 7,
    KanjiNumeric = 1 << 8,
    Greek = 1 << 9,
    Cyrillic = 1 << 10,
    User1 = 1 << 11,
    User2 = 1 << 12,
    User3 = 1 << 13,
    User4 = 1 << 14,
    NoOovBow = 1u << 30,
    NoOovEow = 1u << 31,
    All = 0b00111111_11111111_11111111_11111111,
}

/// <summary>
/// A Sudachi system dictionary (the 0.7 binary format: a description of named blocks — the connection matrix, the parts of
/// speech, a double-array trie over UTF-8 bytes, delta-coded word pointers, the entries and their UTF-16 strings) with the
/// resources the default configuration reads (<c>char.def</c>, <c>unk.def</c>, <c>rewrite.def</c>). Ported from
/// WorksApplications/sudachi.rs 0.7.0 (Apache 2.0); the dictionary and resources are read at runtime, not embedded.
/// </summary>
internal sealed class SudachiDictionary
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("SudachiBinaryDic");
    private readonly byte[] _data;
    private readonly int _connOffset;
    private readonly int _wordPointers;
    private readonly int _entries;
    private readonly int _strings;
    private readonly uint[] _trie;

    public int NumLeft { get; }
    public int NumRight { get; }
    public List<string[]> PartsOfSpeech { get; } = new();
    public SudachiCharacterCategory Categories { get; }

    /// <summary>MeCab-style unknown-word settings per category (<c>char.def</c>'s properties, <c>unk.def</c>).</summary>
    internal Dictionary<SudachiCategory, (bool Invoke, bool Group, int Length)> CategoryProperties { get; } = new();
    internal Dictionary<SudachiCategory, List<(short Left, short Right, short Cost, int Pos)>> UnknownWords { get; } = new();

    /// <summary><c>rewrite.def</c>: characters left unnormalized, and replacements.</summary>
    internal HashSet<int> IgnoreNormalize { get; } = new();
    internal Dictionary<string, string> Replacements { get; } = new(StringComparer.Ordinal);
    internal int LongestReplacementKey { get; private set; }

    private SudachiDictionary(byte[] data, string resources)
    {
        _data = data;
        int position = 0;
        if (data.Length < 32 || !data.Take(16).SequenceEqual(Magic))
            throw new InvalidDataException("Not a Sudachi 0.7 dictionary (SudachiBinaryDic); older formats are not supported.");
        position = 16;
        ulong version = ReadU64(ref position);
        if (version != 1) throw new InvalidDataException($"Unsupported Sudachi dictionary version {version}.");
        position += 16;                                                                  // creation time, flags
        ReadUtf8(ref position);                                                          // comment
        ReadUtf8(ref position);                                                          // signature
        if (ReadUtf8(ref position).Length != 0) throw new InvalidDataException("This is a user dictionary; give the system dictionary.");
        ReadVarint(ref position);                                                        // indexed entries
        ReadVarint(ref position);                                                        // total entries
        var blocks = new Dictionary<string, (int Start, int Size)>(StringComparer.Ordinal);
        long count = ReadVarint(ref position);
        for (long i = 0; i < count; i++)
        {
            string name = ReadUtf8(ref position);
            long start = ReadVarint(ref position), size = ReadVarint(ref position);
            if (start + size > data.Length) throw new InvalidDataException($"Dictionary block {name} lies outside the file.");
            blocks[name] = ((int)start, (int)size);
        }
        (int Start, int Size) Block(string name) => blocks.TryGetValue(name, out var b) ? b : throw new InvalidDataException($"The dictionary has no {name} block.");

        var conn = Block("ConnMatrix");
        NumLeft = BitConverter.ToInt16(data, conn.Start);
        NumRight = BitConverter.ToInt16(data, conn.Start + 2);
        _connOffset = conn.Start + 4;
        var pos = Block("POS");
        int p = pos.Start;
        int posCount = BitConverter.ToInt16(data, p);
        p += 2;
        for (int i = 0; i < posCount; i++)
        {
            var entry = new string[6];
            for (int j = 0; j < 6; j++)
            {
                int length = BitConverter.ToInt16(data, p);
                entry[j] = Encoding.Unicode.GetString(data, p + 2, 2 * length);
                p += 2 + 2 * length;
            }
            PartsOfSpeech.Add(entry);
        }
        var trie = Block("TrieIndex");
        _trie = new uint[trie.Size / 4];
        Buffer.BlockCopy(data, trie.Start, _trie, 0, _trie.Length * 4);
        _wordPointers = Block("WordPointers").Start;
        _entries = Block("Entries").Start;
        _strings = Block("Strings").Start;

        Categories = SudachiCharacterCategory.Read(Path.Combine(resources, "char.def"));
        ReadCategoryProperties(Path.Combine(resources, "char.def"));
        ReadUnknownWords(Path.Combine(resources, "unk.def"));
        ReadRewrites(Path.Combine(resources, "rewrite.def"));
    }

    /// <summary>Opens the system dictionary <paramref name="systemDic"/> with the resources in <paramref name="resources"/>
    /// (SudachiPy's <c>resources</c> directory).</summary>
    public static SudachiDictionary Open(string systemDic, string resources)
    {
        foreach (var file in new[] { systemDic, Path.Combine(resources, "char.def"), Path.Combine(resources, "unk.def"), Path.Combine(resources, "rewrite.def") })
            if (!File.Exists(file)) throw new FileNotFoundException($"Sudachi needs {file}.", file);
        return new SudachiDictionary(File.ReadAllBytes(systemDic), resources);
    }

    private ulong ReadU64(ref int p)
    {
        ulong v = BitConverter.ToUInt64(_data, p);
        p += 8;
        return v;
    }

    private long ReadVarint(ref int p) => ReadVarint(_data, ref p);

    internal static long ReadVarint(byte[] data, ref int p)
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            byte b = data[p++];
            value |= (ulong)(b & 0x7f) << shift;
            if (b < 0x80) return (long)value;
        }
        throw new InvalidDataException("Invalid varint in the Sudachi dictionary.");
    }

    private string ReadUtf8(ref int p)
    {
        int length = (int)ReadVarint(ref p);
        string s = Encoding.UTF8.GetString(_data, p, length);
        p += length;
        return s;
    }

    /// <summary>The connection cost between a left node's right id and a right node's left id.</summary>
    public short Cost(int rightIdOfLeft, int leftIdOfRight) =>
        BitConverter.ToInt16(_data, _connOffset + 2 * (leftIdOfRight * NumLeft + rightIdOfLeft));

    /// <summary>The entries whose surface is a prefix of <paramref name="input"/> from byte <paramref name="offset"/>, in the
    /// trie's order: (entry id, end byte).</summary>
    public IEnumerable<(int Entry, int End)> Lookup(byte[] input, int offset)
    {
        int node = Offset(_trie[0]);
        for (int i = offset; i < input.Length; i++)
        {
            node ^= input[i];
            uint unit = _trie[node];
            if ((unit & ((1u << 31) | 0xFF)) != input[i]) yield break;
            node ^= Offset(unit);
            if (((unit >> 8) & 1) == 1)
            {
                int pointer = (int)(_trie[node] & ((1u << 31) - 1));
                int q = _wordPointers + pointer;
                long n = ReadVarint(ref q);
                int sum = 0;
                for (long k = 0; k < n; k++)
                {
                    sum += (int)ReadVarint(ref q);
                    yield return (sum, i + 1);
                }
            }
        }
    }

    private static int Offset(uint unit) => (int)((unit >> 10) << (int)((unit & (1u << 9)) >> 6));

    private int EntryOffset(int entry) => _entries + (entry << 3);

    public (short Left, short Right, short Cost) Parameters(int entry)
    {
        int o = EntryOffset(entry);
        return (BitConverter.ToInt16(_data, o), BitConverter.ToInt16(_data, o + 2), BitConverter.ToInt16(_data, o + 4));
    }

    public int PosId(int entry) => BitConverter.ToInt16(_data, EntryOffset(entry) + 6);

    private string String(uint encoded)
    {
        // StringPointer.decode.
        uint baseValue = encoded >> 27;
        uint extraBits = baseValue > 19 ? baseValue - 19 : 0;
        uint nonFixed = (encoded & 0x07ff_0000) >> (int)(16 + 12 - extraBits);
        uint implicitBit = (1u << 12) >> (int)(13 - extraBits);
        uint length = baseValue - extraBits + (nonFixed | implicitBit);
        int alignment = (int)(extraBits > 0 ? extraBits - 1 : 0);
        uint offset = (encoded & (0x07ff_ffffu >> alignment)) << alignment;
        return Encoding.Unicode.GetString(_data, _strings + 2 * (int)offset, 2 * (int)length);
    }

    public string Headword(int entry) => String(BitConverter.ToUInt32(_data, EntryOffset(entry) + 8));
    public string Reading(int entry) => String(BitConverter.ToUInt32(_data, EntryOffset(entry) + 12));

    /// <summary>The normalized form: the headword, or the referenced entry's headword.</summary>
    public string NormalizedForm(int entry)
    {
        uint reference = BitConverter.ToUInt32(_data, EntryOffset(entry) + 16) & 0x0fff_ffff;
        return (int)reference == entry ? Headword(entry) : Headword((int)reference);
    }

    /// <summary>The A-unit split: the entries this one splits into in mode A (empty or one: no split).</summary>
    public int[] SplitA(int entry)
    {
        int o = EntryOffset(entry);
        int c = (sbyte)_data[o + 26], b = (sbyte)_data[o + 27], a = (sbyte)_data[o + 28];
        int arrays = o + 32;
        int[] Read(int at, int count) => Enumerable.Range(0, count).Select(i => (int)(BitConverter.ToUInt32(_data, at + 4 * i) & 0x0fff_ffff)).ToArray();
        var cSplit = Read(arrays, c);
        int bAt = arrays + 4 * c;
        var bSplit = b < 0 ? cSplit : Read(bAt, b);
        int aAt = bAt + 4 * Math.Max(0, b);
        return a < 0 ? bSplit : Read(aAt, a);
    }

    public int PartOfSpeechId(params string[] pos)
    {
        for (int i = 0; i < PartsOfSpeech.Count; i++)
            if (PartsOfSpeech[i].SequenceEqual(pos)) return i;
        throw new InvalidDataException($"The dictionary has no part of speech {string.Join(",", pos)}.");
    }

    internal static SudachiCategory ParseCategory(string name) => name switch
    {
        "DEFAULT" => SudachiCategory.Default, "SPACE" => SudachiCategory.Space, "KANJI" => SudachiCategory.Kanji,
        "SYMBOL" => SudachiCategory.Symbol, "NUMERIC" => SudachiCategory.Numeric, "ALPHA" => SudachiCategory.Alpha,
        "HIRAGANA" => SudachiCategory.Hiragana, "KATAKANA" => SudachiCategory.Katakana, "KANJINUMERIC" => SudachiCategory.KanjiNumeric,
        "GREEK" => SudachiCategory.Greek, "CYRILLIC" => SudachiCategory.Cyrillic, "USER1" => SudachiCategory.User1,
        "USER2" => SudachiCategory.User2, "USER3" => SudachiCategory.User3, "USER4" => SudachiCategory.User4,
        "NOOOVBOW" => SudachiCategory.NoOovBow, "NOOOVEOW" => SudachiCategory.NoOovEow, "ALL" => SudachiCategory.All,
        _ => throw new InvalidDataException($"Unknown Sudachi character category '{name}'."),
    };

    private void ReadCategoryProperties(string path)
    {
        foreach (var raw in File.ReadLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#') || line.StartsWith("0x", StringComparison.Ordinal)) continue;
            var cols = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (cols.Length < 4) throw new InvalidDataException($"Malformed char.def line '{line}'.");
            CategoryProperties[ParseCategory(cols[0])] = (cols[1] == "1", cols[2] == "1", int.Parse(cols[3], CultureInfo.InvariantCulture));
        }
    }

    private void ReadUnknownWords(string path)
    {
        foreach (var raw in File.ReadLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var cols = line.Split(',');
            if (cols.Length < 10) throw new InvalidDataException($"Malformed unk.def line '{line}'.");
            var category = ParseCategory(cols[0]);
            if (!CategoryProperties.ContainsKey(category)) throw new InvalidDataException($"{cols[0]} is undefined in char.def.");
            var word = (short.Parse(cols[1], CultureInfo.InvariantCulture), short.Parse(cols[2], CultureInfo.InvariantCulture),
                short.Parse(cols[3], CultureInfo.InvariantCulture), PartOfSpeechId(cols.Skip(4).Take(6).ToArray()));
            if (!UnknownWords.TryGetValue(category, out var list)) UnknownWords[category] = list = new();
            list.Add(word);
        }
    }

    private void ReadRewrites(string path)
    {
        foreach (var raw in File.ReadLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var cols = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (cols.Length == 1)
            {
                var runes = cols[0].EnumerateRunes().ToArray();
                if (runes.Length != 1) throw new InvalidDataException($"{cols[0]} is not a character (rewrite.def).");
                IgnoreNormalize.Add(runes[0].Value);
            }
            else if (cols.Length == 2)
            {
                if (Replacements.ContainsKey(cols[0])) throw new InvalidDataException($"{cols[0]} is already defined (rewrite.def).");
                Replacements[cols[0]] = cols[1];
                LongestReplacementKey = Math.Max(LongestReplacementKey, cols[0].EnumerateRunes().Count());
            }
            else throw new InvalidDataException($"Malformed rewrite.def line '{line}'.");
        }
    }
}

/// <summary>Sudachi's character categories from <c>char.def</c> (<c>CharacterCategory</c>): ranges of code points, the
/// categories of overlapping ranges combined, uncovered code points DEFAULT.</summary>
internal sealed class SudachiCharacterCategory
{
    private readonly uint[] _boundaries;
    private readonly SudachiCategory[] _categories;

    private SudachiCharacterCategory(uint[] boundaries, SudachiCategory[] categories)
    {
        _boundaries = boundaries;
        _categories = categories;
    }

    public static SudachiCharacterCategory Read(string path)
    {
        var ranges = new List<(uint Begin, uint End, SudachiCategory Categories)>();
        foreach (var raw in File.ReadLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#') || !line.StartsWith("0x", StringComparison.Ordinal)) continue;
            var cols = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (cols.Length < 2) throw new InvalidDataException($"Malformed char.def line '{line}'.");
            var r = cols[0].Split(new[] { ".." }, StringSplitOptions.None);
            uint begin = uint.Parse(r[0].Substring(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            uint end = r.Length > 1 ? uint.Parse(r[1].Substring(2), NumberStyles.HexNumber, CultureInfo.InvariantCulture) + 1 : begin + 1;
            var categories = SudachiCategory.None;
            foreach (var element in cols.Skip(1).TakeWhile(e => !e.StartsWith('#'))) categories |= SudachiDictionary.ParseCategory(element);
            ranges.Add((begin, end, categories));
        }
        if (ranges.Count == 0) return new SudachiCharacterCategory(Array.Empty<uint>(), new[] { SudachiCategory.Default });
        // compile(): each segment [b(i−1), b(i)) takes the categories of every range covering it.
        var boundaries = new SortedSet<uint>(ranges.SelectMany(x => new[] { x.Begin, x.End })).ToArray();
        var categoriesBySegment = new SudachiCategory[boundaries.Length];
        foreach (var range in ranges)
        {
            int start = Array.BinarySearch(boundaries, range.Begin) + 1;
            for (int i = start; i < boundaries.Length && boundaries[i] <= range.End; i++) categoriesBySegment[i] |= range.Categories;
        }
        categoriesBySegment[0] = SudachiCategory.Default;
        var finalBoundaries = new List<uint>();
        var finalCategories = new List<SudachiCategory>();
        var lastCategory = categoriesBySegment[0];
        uint lastBoundary = boundaries[0];
        for (int i = 1; i < categoriesBySegment.Length; i++)
        {
            if (categoriesBySegment[i] == lastCategory)
            {
                lastBoundary = boundaries[i];
                continue;
            }
            finalBoundaries.Add(lastBoundary);
            finalCategories.Add(lastCategory);
            lastCategory = categoriesBySegment[i];
            lastBoundary = boundaries[i];
        }
        finalCategories.Add(lastCategory);
        finalBoundaries.Add(lastBoundary);
        for (int i = 0; i < finalCategories.Count; i++)
            if (finalCategories[i] == SudachiCategory.None) finalCategories[i] = SudachiCategory.Default;
        finalCategories.Add(SudachiCategory.Default);
        return new SudachiCharacterCategory(finalBoundaries.ToArray(), finalCategories.ToArray());
    }

    public SudachiCategory Of(int codePoint)
    {
        if (_boundaries.Length == 0) return SudachiCategory.Default;
        int index = Array.BinarySearch(_boundaries, (uint)codePoint);
        return index >= 0 ? _categories[index + 1] : _categories[~index];
    }
}
