using System.Globalization;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// pykakasi 2.3.0's <c>kakasi().convert(text)</c> (kakasi.py), the segments' original text and katakana: the text
/// cut where its script changes (kanji words read through the kanwa dictionary, the longest key whose reading applies
/// after the preceding word, variant kanji normalized first), each segment's reading in katakana. Its quirks are kept:
/// an unknown kanji yields an empty reading and also skips the character after it, private-use characters are dropped,
/// and the Greek range ends at U+30A1 (<c>Ch.greece_Rho</c>).
/// </summary>
/// <remarks>The dictionary is 2.2.1's export with 2.3.0's changes applied (tools/reference-data/pykakasi230_resources.py).</remarks>
internal sealed class Pykakasi
{
    private static readonly Lazy<Pykakasi> SharedInstance = new(() => new Pykakasi(), isThreadSafe: true);

    /// <summary>The shared instance (the dictionary loads once, on first use).</summary>
    public static Pykakasi Default => SharedInstance.Value;

    private sealed class Entry
    {
        public int[] Key = Array.Empty<int>();
        public (string Yomi, string[]? Context)[] Readings = Array.Empty<(string, string[]?)>();
    }

    private readonly Dictionary<int, Dictionary<string, Entry>> _kanwa = new();
    private readonly Dictionary<int, int[]> _itaiji = new();

    private Pykakasi()
    {
        void Add(string key, (string, string[]?)[] readings)
        {
            var codes = CodePoints(key);
            if (!_kanwa.TryGetValue(codes[0], out var table)) _kanwa[codes[0]] = table = new Dictionary<string, Entry>(StringComparer.Ordinal);
            table[key] = new Entry { Key = codes, Readings = readings };
        }
        foreach (var line in ChineseText.ReadLines("pykakasi221_kanwa.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            Add(line.Substring(0, tab), line.Substring(tab + 1).Split('\x1f').Select(r => (r, (string[]?)null)).ToArray());
        }
        foreach (var line in ChineseText.ReadLines("pykakasi230_kanwa_delta.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            if (tab < 0)
            {
                // an entry 2.3.0 removes
                var codes = CodePoints(line);
                _kanwa[codes[0]].Remove(line);
                continue;
            }
            Add(line.Substring(0, tab), line.Substring(tab + 1).Split('\x1f').Select(r =>
            {
                int rs = r.IndexOf('\x1e');
                return rs < 0 ? (r, (string[]?)null) : (r.Substring(0, rs), r.Substring(rs + 1).Split('\x1d'));
            }).ToArray());
        }
        foreach (var line in ChineseText.ReadLines("pykakasi221_itaiji.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            _itaiji[int.Parse(line.Substring(0, tab), CultureInfo.InvariantCulture)] = CodePoints(line.Substring(tab + 1));
        }
    }

    private static int[] CodePoints(string s)
    {
        var result = new List<int>(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            result.Add(char.ConvertToUtf32(s, i));
            if (char.IsHighSurrogate(s[i])) i++;
        }
        return result.ToArray();
    }

    private static string Text(IEnumerable<int> codePoints) => string.Concat(codePoints.Select(char.ConvertFromUtf32));

    // ------------------------------------------------------------------ character classes (properties.py, scripts.py)

    private const string EndMark = ")]!,.,、。？０＞＜";
    private const string LongSymbols = "ー―−ｰ";

    private static bool IsKatakana(int c) => (c > 0x30A0 && c < 0x30FD) || (c > 0xFF65 && c < 0xFF9F) || (c >= 0x1B164 && c <= 0x1B167);
    private static bool IsHiragana(int c) => (c > 0x3040 && c < 0x3097) || (c >= 0x1B150 && c <= 0x1B152);
    private static bool IsAlpha(int c) => c >= 0x20 && c < 0x7F;

    private static bool IsSymbol(int c) =>
        (c >= 0x3000 && c <= 0x3020) || (c >= 0x3030 && c <= 0x303F) || (c >= 0x0391 && c <= 0x30A1)
        || (c >= 0x30A3 && c <= 0x03A9) || (c >= 0x03B1 && c <= 0x03C9) || (c >= 0x0410 && c <= 0x044F)
        || (c >= 0xFF01 && c <= 0xFF1A) || (c >= 0x00A1 && c <= 0x00FF) || (c >= 0xFF20 && c <= 0xFF5E)
        || c == 0x0451 || c == 0x0401;

    private bool IsKanji(int c) => (c >= 0x3400 && c < 0xE000) || _itaiji.ContainsKey(c);

    private static bool IsVariationSelector(int c) => (c >= 0x0E0100 && c <= 0x0E1EF) || (c >= 0xFE00 && c <= 0xFE02);

    // ------------------------------------------------------------------ JConv.convert

    private (string Yomi, int Length) KanjiReading(int[] itext, int start, string btext)
    {
        var text = new List<int>(itext.Length - start);
        for (int i = start; i < itext.Length; i++)
        {
            if (_itaiji.TryGetValue(itext[i], out var replacement)) text.AddRange(replacement);
            else text.Add(itext[i]);
        }
        int length = itext.Length - start;
        int variationSelectors = length - text.Count;
        if (text.Count == 0) throw new IndexOutOfRangeException("string index out of range");
        if (!_kanwa.TryGetValue(text[0], out var table)) return ("", 0);
        int maxLength = 0;
        string yomi = "";
        foreach (var entry in table.Values)
        {
            int keyLength = entry.Key.Length;
            if (text.Count < keyLength) continue;
            bool prefix = true;
            for (int k = 0; k < keyLength && prefix; k++) prefix = text[k] == entry.Key[k];
            if (!prefix) continue;
            foreach (var (reading, context) in entry.Readings)
                if ((context is null || context.Contains(btext)) && maxLength < keyLength)
                {
                    yomi = reading;
                    maxLength = keyLength;
                }
        }
        // Python indexing: [-1] is the last element.
        int At(IReadOnlyList<int> list, int index) => list[index < 0 ? list.Count + index : index];
        var original = new ArraySegment<int>(itext, start, length);
        for (int v = 0; v < variationSelectors; v++)
        {
            if (maxLength > length) break;
            if (At(text, maxLength - 1) != At(original, maxLength - 1)) maxLength++;
            else if (maxLength < variationSelectors + text.Count && maxLength <= length && IsVariationSelector(At(original, maxLength))) maxLength++;
        }
        return (yomi, maxLength);
    }

    // IConv.convert's "kana": the reading's hiragana in katakana.
    private static string ToKatakana(string hira)
    {
        var codes = CodePoints(hira);
        for (int i = 0; i < codes.Length; i++)
        {
            if (codes[i] > 0x3040 && codes[i] < 0x3097) codes[i] += 0x30A1 - 0x3041;
            else if (codes[i] >= 0x1B150 && codes[i] <= 0x1B152) codes[i] += 0x1B164 - 0x1B150;
        }
        return Text(codes);
    }

    private enum Script { Kanji, Kana, Hiragana, Symbol, Alpha }

    /// <summary><c>kakasi().convert(text)</c>: each segment's original text and katakana reading.</summary>
    public List<(string Orig, string Kana)> Convert(string input)
    {
        if (input.Length == 0) return new List<(string, string)> { ("", "") };
        var text = CodePoints(input);
        var result = new List<(string, string)>();
        var original = new StringBuilder();
        var kana = new StringBuilder();
        void Flush() => result.Add((original.ToString(), ToKatakana(kana.ToString())));
        string Char(int index) => char.ConvertFromUtf32(text[index]);

        var previous = Script.Kanji;
        int i = 0;
        while (i < text.Length)
        {
            int c = text[i];
            string action;
            if (c < 0x10000 && EndMark.IndexOf((char)c) >= 0)
            {
                previous = Script.Symbol;
                action = "bufout";
            }
            else if (c < 0x10000 && LongSymbols.IndexOf((char)c) >= 0)
            {
                action = "put";
            }
            else if (IsKatakana(c))
            {
                action = previous == Script.Kana ? "put" : "bufout";
                previous = Script.Kana;
            }
            else if (IsHiragana(c))
            {
                action = previous == Script.Hiragana ? "put" : "bufout";
                previous = Script.Hiragana;
            }
            else if (IsAlpha(c))
            {
                action = previous == Script.Alpha ? "put" : "bufout";
                previous = Script.Alpha;
            }
            else if (IsSymbol(c))
            {
                action = previous != Script.Symbol ? "bufout" : "output";
                previous = Script.Symbol;
            }
            else if (IsKanji(c))
            {
                if (original.Length > 0) Flush();
                var (yomi, length) = KanjiReading(text, i, original.ToString());
                previous = Script.Kanji;
                if (length > 0)
                {
                    original.Clear().Append(Text(text.Skip(i).Take(length)));
                    kana.Clear().Append(yomi);
                    i += length;
                    action = "none";
                }
                else
                {
                    // unknown kanji
                    original.Clear().Append(Char(i));
                    kana.Clear();
                    i++;
                    action = "skip";
                }
            }
            else if ((c >= 0xF000 && c <= 0xFFFD) || (c >= 0x10000 && c <= 0x10FFFD))
            {
                // PUA: ignore and drop
                previous = Script.Symbol;
                if (original.Length > 0) Flush();
                i++;
                action = "none";
            }
            else
            {
                if (original.Length > 0) Flush();
                result.Add((Char(i), ""));
                i++;
                action = "none";
            }

            switch (action)
            {
                case "bufout":
                    if (original.Length > 0) Flush();
                    original.Clear().Append(Char(i));
                    kana.Clear().Append(Char(i));
                    i++;
                    break;
                case "put":
                    original.Append(Char(i));
                    kana.Append(Char(i));
                    i++;
                    break;
                case "output":
                    if (original.Length > 0) Flush();
                    result.Add((Char(i), ToKatakana(Char(i))));
                    original.Clear();
                    kana.Clear();
                    i++;
                    break;
                case "skip":
                    if (original.Length > 0) Flush();
                    original.Clear();
                    kana.Clear();
                    i++;
                    break;
            }
        }
        if (original.Length > 0) Flush();
        return result;
    }
}
