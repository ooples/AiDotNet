using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// A read-only reader of SQLite 3 database files (the documented file format: B-tree pages, records, overflow chains), for
/// front ends whose reference data ships as a SQLite database. It finds rows by rowid and index entries by a key prefix,
/// in the order SQLite's own index scans return them.
/// </summary>
/// <remarks>Only UTF-8 databases with the default BINARY collation are read; text compares as SQLite compares it there,
/// byte by byte.</remarks>
internal sealed class SqliteFile : IDisposable
{
    private readonly FileStream _file;
    private readonly int _pageSize, _usable;
    private readonly Dictionary<string, (string Type, string Table, int Root, string Sql)> _schema = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<int, byte[]> _cache = new();
    private readonly Queue<int> _cacheOrder = new();
    private const int CachePages = 4096;

    private SqliteFile(FileStream file)
    {
        _file = file;
        var header = new byte[100];
        ReadExactly(0, header);
        if (Encoding.ASCII.GetString(header, 0, 15) != "SQLite format 3") throw new InvalidDataException("Not a SQLite 3 database.");
        int size = (header[16] << 8) | header[17];
        _pageSize = size == 1 ? 65536 : size;
        _usable = _pageSize - header[20];
        int encoding = (header[56] << 24) | (header[57] << 16) | (header[58] << 8) | header[59];
        if (encoding != 1) throw new NotSupportedException("Only UTF-8 SQLite databases are read.");
        foreach (var (_, row) in Scan(1))
        {
            string type = (string)row[0]!, name = (string)row[1]!;
            _schema[name] = (type, (string)row[2]!, (int)(long)(row[3] ?? 0L), row[4] as string ?? "");
        }
    }

    /// <summary>Opens a database file for reading.</summary>
    public static SqliteFile Open(string path) =>
        new(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.RandomAccess));

    public void Dispose() => _file.Dispose();

    private void ReadExactly(long position, byte[] buffer)
    {
        lock (_file)
        {
            _file.Seek(position, SeekOrigin.Begin);
            for (int read = 0; read < buffer.Length;)
            {
                int n = _file.Read(buffer, read, buffer.Length - read);
                if (n <= 0) throw new InvalidDataException("The SQLite file ended inside a page.");
                read += n;
            }
        }
    }

    private byte[] Page(int number)
    {
        lock (_cache)
            if (_cache.TryGetValue(number, out var cached)) return cached;
        var page = new byte[_pageSize];
        ReadExactly((long)(number - 1) * _pageSize, page);
        lock (_cache)
        {
            if (_cache.Count >= CachePages) _cache.Remove(_cacheOrder.Dequeue());
            if (!_cache.ContainsKey(number))
            {
                _cache[number] = page;
                _cacheOrder.Enqueue(number);
            }
        }
        return page;
    }

    private static int U16(byte[] b, int at) => (b[at] << 8) | b[at + 1];
    private static int U32(byte[] b, int at) => (b[at] << 24) | (b[at + 1] << 16) | (b[at + 2] << 8) | b[at + 3];

    private static long Varint(byte[] b, ref int at)
    {
        long v = 0;
        for (int i = 0; i < 8; i++)
        {
            byte c = b[at++];
            v = (v << 7) | (long)(c & 0x7F);
            if ((c & 0x80) == 0) return v;
        }
        return (v << 8) | b[at++];
    }

    // A cell's payload: the local part, then the overflow chain.
    private byte[] Payload(byte[] page, int at, long total, bool table)
    {
        int u = _usable;
        long maxLocal = table ? u - 35 : (u - 12) * 64 / 255 - 23;
        long minLocal = (u - 12) * 32 / 255 - 23;
        long local = total;
        if (total > maxLocal)
        {
            long k = minLocal + (total - minLocal) % (u - 4);
            local = k <= maxLocal ? k : minLocal;
        }
        var payload = new byte[total];
        Array.Copy(page, at, payload, 0, local);
        if (local < total)
        {
            int next = U32(page, at + (int)local);
            long done = local;
            while (done < total)
            {
                var overflow = Page(next);
                next = U32(overflow, 0);
                int n = (int)Math.Min(u - 4, total - done);
                Array.Copy(overflow, 4, payload, done, n);
                done += n;
            }
        }
        return payload;
    }

    // A record's values: null, long, double, string or byte[].
    private static object?[] Record(byte[] payload)
    {
        int at = 0;
        long headerSize = Varint(payload, ref at);
        var types = new List<long>();
        while (at < headerSize) types.Add(Varint(payload, ref at));
        var values = new object?[types.Count];
        int body = (int)headerSize;
        for (int i = 0; i < types.Count; i++)
        {
            long t = types[i];
            switch (t)
            {
                case 0: values[i] = null; break;
                case 1: values[i] = (long)(sbyte)payload[body]; body += 1; break;
                case 2: values[i] = (long)(short)((payload[body] << 8) | payload[body + 1]); body += 2; break;
                case 3:
                {
                    int v = (payload[body] << 16) | (payload[body + 1] << 8) | payload[body + 2];
                    values[i] = (long)((v << 8) >> 8);
                    body += 3;
                    break;
                }
                case 4: values[i] = (long)U32(payload, body); body += 4; break;
                case 5:
                {
                    long v = 0;
                    for (int k = 0; k < 6; k++) v = (v << 8) | payload[body + k];
                    values[i] = (v << 16) >> 16;
                    body += 6;
                    break;
                }
                case 6:
                case 7:
                {
                    long v = 0;
                    for (int k = 0; k < 8; k++) v = (v << 8) | payload[body + k];
                    values[i] = t == 6 ? v : BitConverter.Int64BitsToDouble(v);
                    body += 8;
                    break;
                }
                case 8: values[i] = 0L; break;
                case 9: values[i] = 1L; break;
                default:
                {
                    int length = (int)((t - (t % 2 == 0 ? 12 : 13)) / 2);
                    if (t % 2 == 0)
                    {
                        var blob = new byte[length];
                        Array.Copy(payload, body, blob, 0, length);
                        values[i] = blob;
                    }
                    else
                    {
                        values[i] = Encoding.UTF8.GetString(payload, body, length);
                    }
                    body += length;
                    break;
                }
            }
        }
        return values;
    }

    // Page header offset: page 1 starts after the 100-byte database header.
    private static int HeaderOffset(int number) => number == 1 ? 100 : 0;

    /// <summary>Every row of the table B-tree at <paramref name="root"/> in rowid order.</summary>
    private IEnumerable<(long RowId, object?[] Values)> Scan(int root)
    {
        var page = Page(root);
        int h = HeaderOffset(root);
        byte type = page[h];
        int cells = U16(page, h + 3);
        if (type == 0x0D)
        {
            for (int i = 0; i < cells; i++)
            {
                int at = U16(page, h + 8 + 2 * i);
                long size = Varint(page, ref at);
                long rowid = Varint(page, ref at);
                yield return (rowid, Record(Payload(page, at, size, table: true)));
            }
            yield break;
        }
        if (type != 0x05) throw new InvalidDataException($"Page {root} is not a table B-tree page.");
        for (int i = 0; i < cells; i++)
            foreach (var row in Scan(U32(page, U16(page, h + 12 + 2 * i)))) yield return row;
        foreach (var row in Scan(U32(page, h + 8))) yield return row;
    }

    /// <summary>The row of <paramref name="table"/> with <paramref name="rowId"/>, or null; an INTEGER PRIMARY KEY column
    /// holds the rowid.</summary>
    public object?[]? Row(string table, long rowId)
    {
        var (_, _, root, sql) = _schema[table];
        int number = root;
        while (true)
        {
            var page = Page(number);
            int h = HeaderOffset(number);
            byte type = page[h];
            int cells = U16(page, h + 3);
            if (type == 0x0D)
            {
                for (int i = 0; i < cells; i++)
                {
                    int at = U16(page, h + 8 + 2 * i);
                    long size = Varint(page, ref at);
                    long id = Varint(page, ref at);
                    if (id != rowId) continue;
                    var values = Record(Payload(page, at, size, table: true));
                    int alias = RowIdAlias(table, sql);
                    if (alias >= 0 && alias < values.Length && values[alias] is null) values[alias] = rowId;
                    return values;
                }
                return null;
            }
            int child = U32(page, h + 8);
            for (int i = 0; i < cells; i++)
            {
                int at = U16(page, h + 12 + 2 * i);
                int left = U32(page, at);
                at += 4;
                if (rowId <= Varint(page, ref at))
                {
                    child = left;
                    break;
                }
            }
            number = child;
        }
    }

    private readonly Dictionary<string, string[]> _columns = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>The columns of <paramref name="table"/> in order, from its CREATE TABLE statement.</summary>
    public string[] Columns(string table)
    {
        lock (_columns)
        {
            if (_columns.TryGetValue(table, out var known)) return known;
            string sql = Regex.Replace(_schema[table].Sql, @"--[^\n]*", "");
            string body = sql.Substring(sql.IndexOf('(') + 1, sql.LastIndexOf(')') - sql.IndexOf('(') - 1);
            var names = new List<string>();
            int depth = 0, start = 0;
            for (int i = 0; i <= body.Length; i++)
            {
                if (i < body.Length && body[i] == '(') depth++;
                else if (i < body.Length && body[i] == ')') depth--;
                else if (i == body.Length || (body[i] == ',' && depth == 0))
                {
                    var words = body.Substring(start, i - start).Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                    if (words.Length > 0 && !Regex.IsMatch(words[0], "^(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)$", RegexOptions.IgnoreCase))
                        names.Add(words[0].Trim('"', '`', '[', ']'));
                    start = i + 1;
                }
            }
            return _columns[table] = names.ToArray();
        }
    }

    private int RowIdAlias(string table, string sql)
    {
        var columns = Columns(table);
        var match = Regex.Match(Regex.Replace(sql, @"--[^\n]*", ""), @"(\w+)\s+INTEGER\s+(NOT\s+NULL\s+)?PRIMARY\s+KEY", RegexOptions.IgnoreCase);
        return match.Success ? Array.FindIndex(columns, c => string.Equals(c, match.Groups[1].Value, StringComparison.OrdinalIgnoreCase)) : -1;
    }

    // SQLite's ordering of values under BINARY collation: NULL < numbers < text < blob; text and blobs byte-wise.
    private static int Compare(object? a, object? b)
    {
        int Rank(object? v) => v switch { null => 0, long or double => 1, string => 2, _ => 3 };
        int ra = Rank(a), rb = Rank(b);
        if (ra != rb) return ra.CompareTo(rb);
        switch (a)
        {
            case null: return 0;
            case long or double: return Convert.ToDouble(a).CompareTo(Convert.ToDouble(b));
            case string s:
            {
                var x = Encoding.UTF8.GetBytes(s);
                var y = Encoding.UTF8.GetBytes((string)b!);
                for (int i = 0; i < Math.Min(x.Length, y.Length); i++)
                    if (x[i] != y[i]) return x[i].CompareTo(y[i]);
                return x.Length.CompareTo(y.Length);
            }
            default:
            {
                var x = (byte[])a;
                var y = (byte[])b!;
                for (int i = 0; i < Math.Min(x.Length, y.Length); i++)
                    if (x[i] != y[i]) return x[i].CompareTo(y[i]);
                return x.Length.CompareTo(y.Length);
            }
        }
    }

    private static int ComparePrefix(object?[] entry, object?[] key)
    {
        for (int i = 0; i < key.Length; i++)
        {
            int c = Compare(entry[i], key[i]);
            if (c != 0) return c;
        }
        return 0;
    }

    /// <summary>The rowids of the entries of index <paramref name="index"/> whose leading columns equal
    /// <paramref name="key"/>, in index order (the key, then rowid).</summary>
    public List<long> Seek(string index, params object?[] key)
    {
        var (type, _, root, _) = _schema[index];
        if (type != "index") throw new ArgumentException($"{index} is not an index.", nameof(index));
        var result = new List<long>();
        Seek(root, key, result);
        return result;
    }

    // In-order walk that skips subtrees wholly below the key and stops at the first entry above it; returns false to stop.
    private bool Seek(int number, object?[] key, List<long> result)
    {
        var page = Page(number);
        int h = HeaderOffset(number);
        byte type = page[h];
        int cells = U16(page, h + 3);
        bool leaf = type == 0x0A;
        if (!leaf && type != 0x02) throw new InvalidDataException($"Page {number} is not an index B-tree page.");
        int pointers = h + (leaf ? 8 : 12);
        for (int i = 0; i < cells; i++)
        {
            int at = U16(page, pointers + 2 * i);
            int left = 0;
            if (!leaf)
            {
                left = U32(page, at);
                at += 4;
            }
            long size = Varint(page, ref at);
            var entry = Record(Payload(page, at, size, table: false));
            int c = ComparePrefix(entry, key);
            if (c < 0) continue;
            if (!leaf && !Seek(left, key, result)) return false;
            if (c > 0) return false;
            result.Add((long)entry[entry.Length - 1]!);
        }
        return leaf || Seek(U32(page, h + 8), key, result);
    }
}
