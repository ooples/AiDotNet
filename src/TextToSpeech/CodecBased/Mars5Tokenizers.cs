using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// minbpe's model file (<c>minbpe v1</c>: the pattern, the special tokens, then one merge per line, numbered from the
/// base vocabulary up), as MARS5 stores its text and speech tokenizers in its checkpoints' metadata.
/// </summary>
internal sealed record MinBpeModel(string Pattern, IReadOnlyList<(string Token, int Id)> Specials, IReadOnlyList<(int Left, int Right)> Merges)
{
    public static MinBpeModel Parse(string text)
    {
        var lines = text.Replace("\r\n", "\n").Split('\n');
        int line = 0;
        if (lines[line++].Trim() != "minbpe v1") throw new InvalidDataException("Not a minbpe v1 model.");
        string pattern = lines[line++].Trim();
        int specials = int.Parse(lines[line++].Trim(), System.Globalization.CultureInfo.InvariantCulture);
        var specialList = new List<(string, int)>();
        for (int i = 0; i < specials; i++)
        {
            var parts = lines[line++].Trim().Split(' ');
            specialList.Add((parts[0], int.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture)));
        }
        var merges = new List<(int, int)>();
        for (; line < lines.Length; line++)
        {
            var parts = lines[line].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) continue;
            merges.Add((int.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture), int.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture)));
        }
        return new MinBpeModel(pattern, specialList, merges);
    }
}

/// <summary>The shared merge loop of minbpe's tokenizers (<c>_encode_chunk</c>: repeatedly merge the adjacent pair
/// with the lowest merge index).</summary>
internal static class MinBpe
{
    public static List<int> EncodeChunk(List<int> ids, Dictionary<(int, int), int> merges)
    {
        while (ids.Count >= 2)
        {
            // min over the pairs in first-seen order, by merge index (unmerged pairs rank last).
            (int, int)? best = null;
            int bestRank = int.MaxValue;
            for (int i = 0; i + 1 < ids.Count; i++)
            {
                var pair = (ids[i], ids[i + 1]);
                if (merges.TryGetValue(pair, out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    best = pair;
                }
            }
            if (best is not { } chosen) break;
            var merged = new List<int>(ids.Count);
            for (int i = 0; i < ids.Count;)
            {
                if (i < ids.Count - 1 && ids[i] == chosen.Item1 && ids[i + 1] == chosen.Item2)
                {
                    merged.Add(bestRank);
                    i += 2;
                }
                else merged.Add(ids[i++]);
            }
            ids = merged;
        }
        return ids;
    }

    public static Dictionary<(int, int), int> MergeTable(MinBpeModel model, int first)
    {
        var table = new Dictionary<(int, int), int>();
        for (int i = 0; i < model.Merges.Count; i++) table[model.Merges[i]] = first + i;
        return table;
    }
}

/// <summary>
/// minbpe's <c>RegexTokenizer</c> (byte-level BPE after GPT-4's split pattern), MARS5's text tokenizer: the text split by
/// its special tokens, each other part by the pattern, each piece's UTF-8 bytes merged.
/// </summary>
internal sealed class MinBpeRegexTokenizer
{
    // GPT-4's split pattern, with the possessive quantifiers of Python's regex module written as atomic groups and
    // Python's \s (which adds U+001C to U+001F).
    private static readonly string S = "[" + PythonText.WhitespaceClass + "]";
    private static readonly string NotS = "[^" + PythonText.WhitespaceClass + @"\p{L}\p{N}]";
    private static readonly Regex Gpt4Split = new(
        @"'(?i:[sdmt]|ll|ve|re)|(?>[^\r\n\p{L}\p{N}]?)\p{L}+|\p{N}{1,3}| ?(?>" + NotS + @"+)[\r\n]*|" + S + @"*[\r\n]|" + S + @"+(?!\S)|" + S + "+");

    private readonly Dictionary<(int, int), int> _merges;
    private readonly Dictionary<string, int> _specials;
    private readonly Regex? _specialPattern;
    private readonly Dictionary<int, byte[]> _vocab = new();

    public MinBpeRegexTokenizer(MinBpeModel model)
    {
        _merges = MinBpe.MergeTable(model, 256);
        _specials = model.Specials.ToDictionary(s => s.Token, s => s.Id, StringComparer.Ordinal);
        if (_specials.Count > 0)
            _specialPattern = new Regex("(" + string.Join("|", _specials.Keys.Select(Regex.Escape)) + ")");
        for (int i = 0; i < 256; i++) _vocab[i] = new[] { (byte)i };
        foreach (var kv in _merges.OrderBy(kv => kv.Value)) _vocab[kv.Value] = _vocab[kv.Key.Item1].Concat(_vocab[kv.Key.Item2]).ToArray();
        foreach (var kv in _specials) _vocab[kv.Value] = Encoding.UTF8.GetBytes(kv.Key);
    }

    /// <summary><c>len(vocab)</c>: the bytes, the merges and the special tokens.</summary>
    public int VocabSize => _vocab.Count;

    public IReadOnlyDictionary<string, int> Specials => _specials;

    /// <summary><c>encode(text, allowed_special="all")</c>.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        var parts = _specialPattern is null ? new[] { text } : _specialPattern.Split(text);
        foreach (var part in parts)
        {
            if (_specials.TryGetValue(part, out int special)) ids.Add(special);
            else ids.AddRange(EncodeOrdinary(part));
        }
        return ids;
    }

    public List<int> EncodeOrdinary(string text)
    {
        var ids = new List<int>();
        foreach (Match m in Gpt4Split.Matches(text))
            ids.AddRange(MinBpe.EncodeChunk(Encoding.UTF8.GetBytes(m.Value).Select(b => (int)b).ToList(), _merges));
        return ids;
    }
}

/// <summary>
/// minbpe's <c>CodebookTokenizer</c>, MARS5's speech tokenizer: BPE over EnCodec codes, a space-separated code string
/// merged as one chunk (codes 0 to 1023 the base vocabulary, each rendered " %04d"); special tokens split it first.
/// </summary>
internal sealed class MinBpeCodebookTokenizer
{
    private readonly Dictionary<(int, int), int> _merges;
    private readonly Dictionary<string, int> _specials;
    private readonly Dictionary<int, string> _inverseSpecials;
    private readonly Dictionary<int, string> _vocab = new();
    private readonly int _codebookSize;

    public MinBpeCodebookTokenizer(MinBpeModel model, int codebookSize = 1024)
    {
        _codebookSize = codebookSize;
        _merges = MinBpe.MergeTable(model, codebookSize);
        _specials = model.Specials.ToDictionary(s => s.Token, s => s.Id, StringComparer.Ordinal);
        _inverseSpecials = _specials.ToDictionary(kv => kv.Value, kv => kv.Key);
        for (int i = 0; i < codebookSize; i++) _vocab[i] = " " + i.ToString("0000", System.Globalization.CultureInfo.InvariantCulture);
        foreach (var kv in _merges.OrderBy(kv => kv.Value)) _vocab[kv.Value] = _vocab[kv.Key.Item1] + _vocab[kv.Key.Item2];
        foreach (var kv in _specials) _vocab[kv.Value] = kv.Key;
    }

    public int VocabSize => _vocab.Count;

    public IReadOnlyDictionary<string, int> Specials => _specials;

    /// <summary><c>encode</c> of codes (as the release builds the string, space-joined, with no special tokens).</summary>
    public List<int> Encode(IReadOnlyList<int> codes)
    {
        if (codes.Any(c => c < 0 || c >= _codebookSize)) throw new ArgumentOutOfRangeException(nameof(codes), $"Codes lie in [0, {_codebookSize}).");
        return codes.Count == 0 ? new List<int>() : MinBpe.EncodeChunk(codes.ToList(), _merges);
    }

    /// <summary><c>decode_int</c>: the codes of token ids, special tokens kept as their strings.</summary>
    public List<object> DecodeInt(IReadOnlyList<int> ids)
    {
        var text = new StringBuilder();
        foreach (int id in ids)
        {
            if (_vocab.TryGetValue(id, out var piece)) text.Append(piece);
            else if (_inverseSpecials.TryGetValue(id, out var special)) text.Append(special);
            else throw new ArgumentException($"Invalid token id {id}.", nameof(ids));
        }
        string decoded = text.ToString();
        foreach (var s in _specials.Keys) decoded = decoded.Replace(s, " " + s + " ");
        return decoded.Trim().Split(' ').Where(t => t.Length > 0)
            .Select(t => char.IsDigit(t[0]) ? (object)int.Parse(t, System.Globalization.CultureInfo.InvariantCulture) : t).ToList();
    }

    /// <summary>The codes among <see cref="DecodeInt"/>'s items (the release keeps the integers).</summary>
    public List<int> DecodeCodes(IReadOnlyList<int> ids) => DecodeInt(ids).OfType<int>().ToList();
}
