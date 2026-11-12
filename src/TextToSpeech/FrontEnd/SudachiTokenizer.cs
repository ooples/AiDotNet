using System.Globalization;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Sudachi's tokenizer with SudachiPy's default configuration (sudachi.rs 0.7.0 <c>StatefulTokenizer</c> and
/// <c>resources/sudachi.json</c>): the input rewritten (DefaultInputTextPlugin with <c>rewrite.def</c>, lower case and NFKC;
/// ProlongedSoundMarkPlugin; IgnoreYomiganaPlugin), a lattice of the dictionary's words and the unknown words of
/// MeCabOovPlugin and SimpleOovPlugin, the cheapest path, JoinNumericPlugin and JoinKatakanaOovPlugin, then the A-unit
/// split. Gives each morpheme's reading form, as Zonos reads them.
/// </summary>
internal sealed class SudachiTokenizer
{
    private static readonly HashSet<int> ProlongedSoundMarks = new() { 'ー', '-', '⁓', '〜', '〰' };
    private static readonly HashSet<int> LeftBrackets = new() { '(', '（' };
    private static readonly HashSet<int> RightBrackets = new() { ')', '）' };
    private const int MaxYomiganaLength = 4;
    private const int MaxInputBytes = ushort.MaxValue / 4 * 3;

    private readonly SudachiDictionary _dic;
    private readonly int _numericPos;
    private readonly int _katakanaOovPos;
    private readonly (short Left, short Right, short Cost, int Pos) _simpleOov;

    public SudachiTokenizer(SudachiDictionary dictionary)
    {
        _dic = dictionary;
        _numericPos = _dic.PartOfSpeechId("名詞", "数詞", "*", "*", "*", "*");
        _katakanaOovPos = _dic.PartOfSpeechId("名詞", "普通名詞", "一般", "*", "*", "*");
        _simpleOov = (5968, 5968, 3857, _dic.PartOfSpeechId("補助記号", "一般", "*", "*", "*", "*"));
    }

    // ---------------------------------------------------------------- input rewriting

    /// <summary>DefaultInputTextPlugin: at each character a <c>rewrite.def</c> replacement starting there, else the
    /// character lower-cased (Unicode's Uppercase property) and NFKC-normalized unless <c>rewrite.def</c> leaves it.</summary>
    internal string Normalize(string text)
    {
        var runes = SudachiText.CodePoints(text);
        var output = new StringBuilder(text.Length);
        for (int i = 0; i < runes.Length; i++)
        {
            bool replaced = false;
            for (int length = Math.Min(_dic.LongestReplacementKey, runes.Length - i); length >= 1 && !replaced; length--)
            {
                string key = string.Concat(runes.Skip(i).Take(length).Select(SudachiText.Of));
                if (_dic.Replacements.TryGetValue(key, out var replacement))
                {
                    output.Append(replacement);
                    i += length - 1;
                    replaced = true;
                }
            }
            if (replaced) continue;
            var rune = runes[i];
            string original = SudachiText.Of(rune);
            string changed = original;
            if (IsUppercase(rune)) changed = Lowercase(rune);
            if (!_dic.IgnoreNormalize.Contains(rune)) changed = changed.Normalize(NormalizationForm.FormKC);
            // handle_normalization_slow: replaced only when the first character changes.
            output.Append(changed.Length > 0 && SudachiText.CodePoints(changed)[0] != rune ? changed : original);
        }
        return output.ToString();
    }

    // Rust's char::is_uppercase: the Uppercase property (Lu and Other_Uppercase).
    private static bool IsUppercase(int c) => CharUnicodeInfo.GetUnicodeCategory(char.ConvertFromUtf32(c), 0) == UnicodeCategory.UppercaseLetter
        || (c >= 0x2160 && c <= 0x216F) || (c >= 0x24B6 && c <= 0x24CF)
        || (c >= 0x1F130 && c <= 0x1F149) || (c >= 0x1F150 && c <= 0x1F169) || (c >= 0x1F170 && c <= 0x1F189);

    private static string Lowercase(int c)
    {
        if (c >= 0x2160 && c <= 0x216F) return char.ConvertFromUtf32(c + 0x10);
        if (c >= 0x24B6 && c <= 0x24CF) return char.ConvertFromUtf32(c + 0x1A);
        if (c == 0x0130) return "i\u0307";
        return char.ConvertFromUtf32(c).ToLowerInvariant();
    }

    /// <summary>ProlongedSoundMarkPlugin: each run of two or more prolonged sound marks as one "ー".</summary>
    internal static string CollapseProlongedSoundMarks(string text)
    {
        var runes = SudachiText.CodePoints(text);
        var output = new StringBuilder(text.Length);
        for (int i = 0; i < runes.Length; i++)
        {
            int j = i;
            while (j < runes.Length && ProlongedSoundMarks.Contains(runes[j])) j++;
            if (j - i >= 2)
            {
                output.Append('ー');
                i = j - 1;
                continue;
            }
            output.Append(SudachiText.Of(runes[i]));
        }
        return output.ToString();
    }

    /// <summary>IgnoreYomiganaPlugin: a kanji's bracketed reading (one to four kana) removed.</summary>
    internal string RemoveYomigana(string text)
    {
        var runes = SudachiText.CodePoints(text);
        var output = new StringBuilder(text.Length);
        for (int i = 0; i < runes.Length; i++)
        {
            output.Append(SudachiText.Of(runes[i]));
            if (!_dic.Categories.Of(runes[i]).HasFlag(SudachiCategory.Kanji) || i + 1 >= runes.Length || !LeftBrackets.Contains(runes[i + 1]))
                continue;
            int reading = 0;
            while (reading <= MaxYomiganaLength && i + 2 + reading < runes.Length
                && (_dic.Categories.Of(runes[i + 2 + reading]) & (SudachiCategory.Hiragana | SudachiCategory.Katakana)) != 0)
                reading++;
            int close = i + 2 + reading;
            if (reading >= 1 && reading <= MaxYomiganaLength && close < runes.Length && RightBrackets.Contains(runes[close]))
                i = close;
        }
        return output.ToString();
    }

    // ---------------------------------------------------------------- the analysis

    private sealed class Input
    {
        public int[] Chars = Array.Empty<int>();
        public byte[] Bytes = Array.Empty<byte>();
        public int[] CharToByte = Array.Empty<int>();
        public int[] ByteToChar = Array.Empty<int>();
        public bool[] BowAtByte = Array.Empty<bool>();
        public SudachiCategory[] Categories = Array.Empty<SudachiCategory>();
        public int[] Continuity = Array.Empty<int>();

        public string Slice(int begin, int end) => string.Concat(Chars.Skip(begin).Take(end - begin).Select(c => SudachiText.Of(c)));

        public SudachiCategory CategoryOfRange(int begin, int end)
        {
            if (begin >= end) return SudachiCategory.None;
            var c = (SudachiCategory)uint.MaxValue;
            for (int i = begin; i < end; i++) c &= Categories[i];
            return c;
        }
    }

    private Input Build(string text)
    {
        var input = new Input { Chars = SudachiText.CodePoints(text) };
        int n = input.Chars.Length;
        input.Bytes = Encoding.UTF8.GetBytes(text);
        input.CharToByte = new int[n + 1];
        input.ByteToChar = new int[input.Bytes.Length + 1];
        input.BowAtByte = new bool[input.Bytes.Length];
        input.Categories = new SudachiCategory[n];
        var nonStarting = SudachiCategory.Alpha | SudachiCategory.Greek | SudachiCategory.Cyrillic;
        var previous = SudachiCategory.None;
        int b = 0;
        for (int i = 0; i < n; i++)
        {
            input.CharToByte[i] = b;
            var category = _dic.Categories.Of(input.Chars[i]);
            input.Categories[i] = category;
            input.BowAtByte[b] = (category & nonStarting) == 0 || (category & previous) == 0;
            previous = category;
            int width = SudachiText.Utf8Length(input.Chars[i]);
            for (int k = 0; k < width; k++) input.ByteToChar[b + k] = i;
            b += width;
        }
        input.CharToByte[n] = b;
        input.ByteToChar[b] = n;
        // fill_cat_continuity: from each character, how far its categories continue in common.
        input.Continuity = new int[n];
        int length = 1;
        for (int start = 0; start < n; start++)
        {
            if (length > 1)
            {
                length--;
                input.Continuity[start] = length;
                continue;
            }
            var common = input.Categories[start];
            length = 1;
            while (start + length < n)
            {
                common &= input.Categories[start + length];
                if (common == SudachiCategory.None) break;
                length++;
            }
            input.Continuity[start] = length;
        }
        return input;
    }

    private bool CanOovBow(Input input, int offset) =>
        !input.Categories[offset].HasFlag(SudachiCategory.NoOovBow) && (offset == 0 || !input.Categories[offset - 1].HasFlag(SudachiCategory.NoOovEow));

    private sealed class Node
    {
        public int Begin, End, Left, Right, Cost, Entry = -1, Pos, Total = int.MaxValue;
        public Node? Previous;
        public bool Oov;

        /// <summary>The forms of an unknown or joined word (null for a dictionary word, which reads them from the
        /// dictionary).</summary>
        public string? Headword, ReadingForm, Normalized;
    }

    // The Viterbi lattice: nodes by end position.
    private sealed class Lattice
    {
        public readonly List<Node>[] Ends;

        public Lattice(int length)
        {
            Ends = new List<Node>[length + 1];
            for (int i = 0; i <= length; i++) Ends[i] = new List<Node>();
            Ends[0].Add(new Node { Total = 0, Right = 0 });
        }

        public void Insert(Node node, SudachiDictionary dic)
        {
            node.Total = int.MaxValue;
            foreach (var left in Ends[node.Begin])
            {
                if (left.Total == int.MaxValue) continue;
                int cost = left.Total + dic.Cost(left.Right, node.Left) + node.Cost;
                if (cost < node.Total)
                {
                    node.Total = cost;
                    node.Previous = left;
                }
            }
            Ends[node.End].Add(node);
        }
    }

    /// <summary>The morphemes' reading forms of <paramref name="text"/> in split mode A.</summary>
    public List<string> Readings(string text)
    {
        string current = RemoveYomigana(CollapseProlongedSoundMarks(Normalize(text ?? throw new ArgumentNullException(nameof(text)))));
        if (Encoding.UTF8.GetByteCount(text) > MaxInputBytes) throw new ArgumentException($"Sudachi reads at most {MaxInputBytes} bytes.", nameof(text));
        if (current.Length == 0) return new List<string>();
        var input = Build(current);
        var path = BestPath(input);
        path = JoinNumeric(input, path);
        path = JoinKatakanaOov(input, path);
        var readings = new List<string>();
        foreach (var node in path)
        {
            if (!node.Oov)
            {
                var split = _dic.SplitA(node.Entry);
                if (split.Length > 1)
                {
                    readings.AddRange(split.Select(e => _dic.Reading(e)));
                    continue;
                }
            }
            readings.Add(ReadingOf(node));
        }
        return readings;
    }

    private string ReadingOf(Node node) => node.ReadingForm ?? _dic.Reading(node.Entry);
    private string HeadwordOf(Node node) => node.Headword ?? _dic.Headword(node.Entry);
    private string NormalizedOf(Node node) => node.Normalized ?? _dic.NormalizedForm(node.Entry);
    private int PosOf(Node node) => node.Entry >= 0 && !node.Oov ? _dic.PosId(node.Entry) : node.Pos;

    private List<Node> BestPath(Input input)
    {
        int n = input.Chars.Length;
        var lattice = new Lattice(n);
        _lattice = lattice;
        for (int offset = 0; offset < n; offset++)
        {
            if (lattice.Ends[offset].Count == 0) continue;
            ulong created = 0;
            foreach (var (entry, endByte) in _dic.Lookup(input.Bytes, input.CharToByte[offset]))
            {
                if (endByte < input.Bytes.Length && !input.BowAtByte[endByte]) continue;
                var (left, right, cost) = _dic.Parameters(entry);
                int end = input.ByteToChar[endByte];
                var node = new Node { Begin = offset, End = end, Left = (ushort)left, Right = (ushort)right, Cost = cost, Entry = entry };
                created = AddWord(created, end - offset);
                lattice.Insert(node, _dic);
            }
            // insert_oovs: every provider where an unknown word may begin; the last one again if nothing was made.
            if (CanOovBow(input, offset))
            {
                created = MeCabOov(input, offset, created, lattice);
                created = SimpleOov(input, offset, created, lattice);
            }
            if (created == 0) created = SimpleOov(input, offset, created, lattice);
            if (created == 0) throw new InvalidOperationException("Sudachi could not connect the beginning to the end.");
        }
        var eos = new Node { Begin = n, End = n, Left = 0, Right = 0, Cost = 0 };
        lattice.Insert(eos, _dic);
        if (eos.Total == int.MaxValue) throw new InvalidOperationException("Sudachi could not connect the beginning to the end.");
        var path = new List<Node>();
        for (var node = eos.Previous; node is not null && node.Previous is not null; node = node.Previous) path.Add(node);
        path.Reverse();
        // Unknown words read as their text.
        foreach (var node in path.Where(x => x.Oov))
        {
            string surface = input.Slice(node.Begin, node.End);
            node.Headword = node.ReadingForm = node.Normalized = surface;
        }
        return path;
    }

    private Lattice? _lattice;

    // CreatedWords: a bit per word length (lengths of 64 and more share the last bit).
    private static ulong AddWord(ulong created, int length) => created | (1UL << Math.Min(Math.Max(length - 1, 0), 63));

    private ulong MeCabOov(Input input, int offset, ulong created, Lattice lattice)
    {
        int charLength = input.Continuity[offset];
        if (charLength == 0) return created;
        ulong other = created;
        // The character's categories one at a time, lowest bit first.
        var categories = input.Categories[offset];
        for (int bit = 0; bit < 32; bit++)
        {
            var category = (SudachiCategory)(1u << bit);
            if ((categories & category) == 0) continue;
            if (!_dic.CategoryProperties.TryGetValue(category, out var info)) continue;
            if (!info.Invoke && other != 0) continue;
            if (!_dic.UnknownWords.TryGetValue(category, out var words)) continue;
            int remaining = charLength;
            if (info.Group)
            {
                foreach (var w in words) created = AddOov(lattice, offset, offset + charLength, w, created);
                remaining--;
            }
            for (int i = 1; i <= info.Length; i++)
            {
                int sublength = Math.Min(offset + i, input.Chars.Length) - offset;
                if (sublength > remaining) break;
                foreach (var w in words) created = AddOov(lattice, offset, offset + sublength, w, created);
            }
        }
        return created;
    }

    private ulong AddOov(Lattice lattice, int begin, int end, (short Left, short Right, short Cost, int Pos) word, ulong created)
    {
        lattice.Insert(new Node { Begin = begin, End = end, Left = (ushort)word.Left, Right = (ushort)word.Right, Cost = word.Cost, Oov = true, Pos = word.Pos }, _dic);
        return AddWord(created, end - begin);
    }

    private ulong SimpleOov(Input input, int offset, ulong created, Lattice lattice)
    {
        if (created != 0) return created;
        // get_word_candidate_length: up to the next character a word may begin at.
        int length = input.Chars.Length - offset;
        for (int i = offset + 1; i < input.Chars.Length; i++)
            if (input.BowAtByte[input.CharToByte[i]])
            {
                length = i - offset;
                break;
            }
        return AddOov(lattice, offset, offset + length, _simpleOov, created);
    }

    // ---------------------------------------------------------------- path rewriting

    // concat_nodes: one node over [begin, end) with the forms concatenated (and the normalized form given).
    private List<Node> Concat(List<Node> path, int begin, int end, string? normalized)
    {
        var node = new Node
        {
            Begin = path[begin].Begin, End = path[end - 1].End, Oov = true, Pos = PosOf(path[begin]), Total = path[end - 1].Total,
            Headword = string.Concat(path.Skip(begin).Take(end - begin).Select(HeadwordOf)),
            ReadingForm = string.Concat(path.Skip(begin).Take(end - begin).Select(ReadingOf)),
            Normalized = normalized ?? string.Concat(path.Skip(begin).Take(end - begin).Select(NormalizedOf)),
        };
        path[begin] = node;
        path.RemoveRange(begin + 1, end - begin - 1);
        return path;
    }

    private List<Node> JoinNumeric(Input input, List<Node> path)
    {
        var parser = new SudachiNumericParser();
        List<Node> Join(List<Node> p, int begin, int end)
        {
            if (PosOf(p[begin]) != _numericPos) return p;
            string normalized = parser.Normalized();
            if (end - begin > 1 || normalized != NormalizedOf(p[begin])) p = Concat(p, begin, end, normalized);
            return p;
        }
        int beginIndex = -1;
        bool commaAsDigit = true, periodAsDigit = true;
        int i = -1;
        while (i < path.Count - 1)
        {
            i++;
            var node = path[i];
            var types = input.CategoryOfRange(node.Begin, node.End);
            string s = NormalizedOf(node);
            if ((types & (SudachiCategory.Numeric | SudachiCategory.KanjiNumeric)) != 0 || (commaAsDigit && s == ",") || (periodAsDigit && s == "."))
            {
                if (beginIndex < 0)
                {
                    parser.Clear();
                    beginIndex = i;
                }
                foreach (var rune in SudachiText.CodePoints(s))
                {
                    if (!parser.Append(rune))
                    {
                        if (beginIndex >= 0)
                        {
                            if (parser.Error == SudachiNumericParser.State.Comma)
                            {
                                commaAsDigit = false;
                                i = beginIndex - 1;
                            }
                            else if (parser.Error == SudachiNumericParser.State.Point)
                            {
                                periodAsDigit = false;
                                i = beginIndex - 1;
                            }
                            beginIndex = -1;
                        }
                        break;
                    }
                }
                continue;
            }
            int c = Encoding.UTF8.GetByteCount(s) == 1 ? s[0] : -1;
            if (beginIndex >= 0)
            {
                if (parser.Done())
                {
                    path = Join(path, beginIndex, i);
                    i = beginIndex + 1;
                }
                else
                {
                    string ss = NormalizedOf(path[i - 1]);
                    if ((parser.Error == SudachiNumericParser.State.Comma && ss == ",") || (parser.Error == SudachiNumericParser.State.Point && ss == "."))
                    {
                        path = Join(path, beginIndex, i - 1);
                        i = beginIndex + 2;
                    }
                }
            }
            beginIndex = -1;
            if (!commaAsDigit && c != ',') commaAsDigit = true;
            if (!periodAsDigit && c != '.') periodAsDigit = true;
        }
        if (beginIndex >= 0)
        {
            int length = path.Count;
            if (parser.Done()) path = Join(path, beginIndex, length);
            else
            {
                string ss = NormalizedOf(path[length - 1]);
                if ((parser.Error == SudachiNumericParser.State.Comma && ss == ",") || (parser.Error == SudachiNumericParser.State.Point && ss == "."))
                    path = Join(path, beginIndex, length - 1);
            }
        }
        return path;
    }

    private List<Node> JoinKatakanaOov(Input input, List<Node> path)
    {
        const int MinLength = 3;
        bool IsKatakana(Node node) => input.CategoryOfRange(node.Begin, node.End).HasFlag(SudachiCategory.Katakana);
        bool CanOovBowNode(Node node) => !input.Categories[node.Begin].HasFlag(SudachiCategory.NoOovBow);
        int i = 0;
        while (i < path.Count)
        {
            var node = path[i];
            if (!(node.Oov || node.End - node.Begin < MinLength) || !IsKatakana(node))
            {
                i++;
                continue;
            }
            int begin = i - 1;
            while (begin >= 0)
            {
                if (!IsKatakana(path[begin]))
                {
                    begin++;
                    break;
                }
                begin--;
            }
            if (begin < 0) begin = 0;
            int end = i + 1;
            while (end < path.Count && IsKatakana(path[end])) end++;
            while (begin != end && !CanOovBowNode(path[begin])) begin++;
            if (end - begin > 1)
            {
                path = ConcatOov(input, path, begin, end);
                i = begin + 1;
            }
            i++;
        }
        return path;
    }

    // concat_oov_nodes: the lattice's cheapest word over the span if there is one, else an unknown word over it.
    private List<Node> ConcatOov(Input input, List<Node> path, int begin, int end)
    {
        int nodeBegin = path[begin].Begin, nodeEnd = path[end - 1].End;
        Node? existing = null;
        foreach (var candidate in _lattice!.Ends[nodeEnd])
            if (candidate.Begin == nodeBegin && (existing is null || candidate.Cost < existing.Cost)) existing = candidate;
        Node replacement;
        if (existing is not null && !(existing.Begin == existing.End))
        {
            replacement = new Node { Begin = existing.Begin, End = existing.End, Entry = existing.Entry, Oov = existing.Oov, Pos = existing.Pos, Total = existing.Total };
            if (existing.Oov)
            {
                string surface = input.Slice(existing.Begin, existing.End);
                replacement.Headword = replacement.ReadingForm = replacement.Normalized = surface;
            }
        }
        else
        {
            string headword = string.Concat(path.Skip(begin).Take(end - begin).Select(HeadwordOf));
            replacement = new Node
            {
                Begin = nodeBegin, End = nodeEnd, Oov = true, Pos = _katakanaOovPos, Total = path[end - 1].Total,
                Headword = headword, ReadingForm = headword, Normalized = headword,
            };
        }
        path[begin] = replacement;
        path.RemoveRange(begin + 1, end - begin - 1);
        return path;
    }
}

/// <summary>Sudachi's numeric parser (JoinNumericPlugin's <c>NumericParser</c> and <c>StringNumber</c>): Arabic and kanji
/// digits with the small units 十百千, the large units 万億兆, commas and a decimal point.</summary>
internal sealed class SudachiNumericParser
{
    internal enum State { None, Point, Comma }

    private sealed class Number
    {
        public StringBuilder Significand = new();
        public int Scale;
        public int Point = -1;
        public bool AllZero = true;

        public void Clear()
        {
            Significand.Clear();
            Scale = 0;
            Point = -1;
            AllZero = true;
        }

        public bool IsZero => Significand.Length == 0;

        public void Append(int digit)
        {
            if (digit != 0) AllZero = false;
            Significand.Append(digit.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        public void ShiftScale(int shift)
        {
            if (IsZero) Significand.Append('1');
            Scale += shift;
        }

        public bool Add(Number other)
        {
            if (other.IsZero) return true;
            if (IsZero)
            {
                Significand.Append(other.Significand);
                Scale = other.Scale;
                Point = other.Point;
                return true;
            }
            NormalizeScale();
            int length = other.IntLength();
            if (Scale >= length)
            {
                Significand.Append('0', Scale - length);
                if (other.Point >= 0) Point = Significand.Length + other.Point;
                Significand.Append(other.Significand);
                Scale = other.Scale;
                return true;
            }
            return false;
        }

        public bool SetPoint()
        {
            if (Scale == 0 && Point < 0)
            {
                Point = Significand.Length;
                return true;
            }
            return false;
        }

        private int IntLength()
        {
            NormalizeScale();
            return Point >= 0 ? Point : Significand.Length + Scale;
        }

        public string Normalized()
        {
            if (IsZero) return "0";
            NormalizeScale();
            if (Scale > 0) Significand.Append('0', Scale);
            else if (Point >= 0)
            {
                Significand.Insert(Point, '.');
                if (Point == 0) Significand.Insert(0, '0');
                int zeros = 0;
                while (zeros < Significand.Length && Significand[Significand.Length - 1 - zeros] == '0') zeros++;
                Significand.Length -= zeros;
                if (Significand.Length > 0 && Significand[Significand.Length - 1] == '.') Significand.Length--;
            }
            return Significand.ToString();
        }

        private void NormalizeScale()
        {
            if (Point < 0) return;
            int nScale = Significand.Length - Point;
            if (nScale > Scale)
            {
                Point += Scale;
                Scale = 0;
            }
            else
            {
                Scale -= nScale;
                Point = -1;
            }
        }
    }

    private static readonly Dictionary<int, int> Values = new()
    {
        ['〇'] = 0, ['一'] = 1, ['二'] = 2, ['三'] = 3, ['四'] = 4, ['五'] = 5, ['六'] = 6, ['七'] = 7, ['八'] = 8, ['九'] = 9,
        ['十'] = -1, ['百'] = -2, ['千'] = -3, ['万'] = -4, ['億'] = -8, ['兆'] = -12,
        ['0'] = 0, ['1'] = 1, ['2'] = 2, ['3'] = 3, ['4'] = 4, ['5'] = 5, ['6'] = 6, ['7'] = 7, ['8'] = 8, ['9'] = 9,
    };

    private int _digitLength;
    private bool _firstDigit = true, _hasComma, _hangingPoint;
    private readonly Number _total = new(), _subtotal = new(), _tmp = new();

    public State Error { get; private set; }

    public void Clear()
    {
        _digitLength = 0;
        _firstDigit = true;
        _hasComma = false;
        _hangingPoint = false;
        Error = State.None;
        _total.Clear();
        _subtotal.Clear();
        _tmp.Clear();
    }

    public bool Append(int c)
    {
        if (c == '.')
        {
            _hangingPoint = true;
            if (_firstDigit)
            {
                Error = State.Point;
                return false;
            }
            if (_hasComma && !CheckComma())
            {
                Error = State.Comma;
                return false;
            }
            if (!_tmp.SetPoint())
            {
                Error = State.Point;
                return false;
            }
            _hasComma = false;
            return true;
        }
        if (c == ',')
        {
            if (!CheckComma())
            {
                Error = State.Comma;
                return false;
            }
            _hasComma = true;
            _digitLength = 0;
            return true;
        }
        if (!Values.TryGetValue(c, out int n)) return false;
        if (n >= -3 && n < 0)
        {
            _tmp.ShiftScale(-n);
            if (!_subtotal.Add(_tmp)) return false;
            _tmp.Clear();
            _firstDigit = true;
            _digitLength = 0;
            _hasComma = false;
        }
        else if (n < -3)
        {
            if (!_subtotal.Add(_tmp) || _subtotal.IsZero) return false;
            _subtotal.ShiftScale(-n);
            if (!_total.Add(_subtotal)) return false;
            _subtotal.Clear();
            _tmp.Clear();
            _firstDigit = true;
            _digitLength = 0;
            _hasComma = false;
        }
        else
        {
            _tmp.Append(n);
            _firstDigit = false;
            _digitLength++;
            _hangingPoint = false;
        }
        return true;
    }

    public bool Done()
    {
        bool result = _subtotal.Add(_tmp) && _total.Add(_subtotal);
        if (_hangingPoint)
        {
            Error = State.Point;
            return false;
        }
        if (_hasComma && _digitLength != 3)
        {
            Error = State.Comma;
            return false;
        }
        return result;
    }

    public string Normalized() => _total.Normalized();

    private bool CheckComma()
    {
        if (_firstDigit) return false;
        if (!_hasComma) return _digitLength <= 3 && !_tmp.IsZero && !_tmp.AllZero;
        return _digitLength == 3;
    }
}
