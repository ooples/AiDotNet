using System.Globalization;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The release's Hugging Face BPE tokenizer, including its added-token handling.</summary>
internal sealed class ChatterboxTokenizer
{
    private static readonly Regex Pieces = new(@"\w+|[^\w\s]+", RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<int, string> _tokens;
    private readonly Dictionary<(string Left, string Right), int> _merges;
    private readonly string[] _addedTokens;
    private readonly int _unknown;

    private ChatterboxTokenizer(
        Dictionary<string, int> vocabulary,
        Dictionary<(string Left, string Right), int> merges,
        IEnumerable<string> addedTokens,
        int unknown)
    {
        _vocabulary = vocabulary;
        _tokens = vocabulary.ToDictionary(pair => pair.Value, pair => pair.Key);
        _merges = merges;
        _addedTokens = addedTokens.OrderByDescending(token => token.Length).ToArray();
        _unknown = unknown;
    }

    public int VocabularySize => _vocabulary.Count;

    public static ChatterboxTokenizer Load(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("The tokenizer path is required.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"The Chatterbox tokenizer was not found: {path}", path);

        var root = JObject.Parse(File.ReadAllText(path));
        var model = root["model"] as JObject
            ?? throw new InvalidDataException("tokenizer.json has no model object.");
        var vocabObject = model["vocab"] as JObject
            ?? throw new InvalidDataException("tokenizer.json has no BPE vocabulary.");
        var vocabulary = vocabObject.Properties().ToDictionary(p => p.Name, p => p.Value.Value<int>(), StringComparer.Ordinal);

        var merges = new Dictionary<(string Left, string Right), int>();
        if (model["merges"] is JArray mergeArray)
        {
            int rank = 0;
            foreach (var item in mergeArray)
            {
                string[] pair;
                if (item is JArray array)
                    pair = array.Values<string>().Where(value => value is not null).Cast<string>().ToArray();
                else
                    pair = (item.Value<string>() ?? string.Empty).Split(new[] { ' ' }, 2);
                if (pair.Length == 2)
                    merges[(pair[0], pair[1])] = rank++;
            }
        }

        var added = new List<string>();
        if (root["added_tokens"] is JArray addedArray)
            foreach (var item in addedArray.OfType<JObject>())
                if (item["content"]?.Value<string>() is { Length: > 0 } token)
                    added.Add(token);

        int unknown = vocabulary.TryGetValue("[UNK]", out int id) ? id : 0;
        return new ChatterboxTokenizer(vocabulary, merges, added, unknown);
    }

    public int[] Encode(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        text = text.Replace(" ", "[SPACE]");
        var ids = new List<int>();
        int start = 0;
        while (start < text.Length)
        {
            string? special = _addedTokens.FirstOrDefault(token =>
                start + token.Length <= text.Length
                && string.CompareOrdinal(text, start, token, 0, token.Length) == 0);
            if (special is not null)
            {
                ids.Add(_vocabulary.TryGetValue(special, out int specialId) ? specialId : _unknown);
                start += special.Length;
                continue;
            }

            int end = start + 1;
            while (end < text.Length && !_addedTokens.Any(token =>
                end + token.Length <= text.Length
                && string.CompareOrdinal(text, end, token, 0, token.Length) == 0))
                end++;

            foreach (Match match in Pieces.Matches(text.Substring(start, end - start)))
                EncodePiece(match.Value, ids);
            start = end;
        }
        return ids.ToArray();
    }

    public string Decode(IEnumerable<int> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        var text = string.Concat(ids.Select(id => _tokens.TryGetValue(id, out var token) ? token : "[UNK]"));
        return text.Replace(" ", string.Empty)
            .Replace("[SPACE]", " ")
            .Replace("[STOP]", string.Empty)
            .Replace("[UNK]", string.Empty);
    }

    private void EncodePiece(string piece, List<int> output)
    {
        var symbols = CodePoints(piece);
        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue;
            (string Left, string Right) best = default;
            for (int i = 0; i + 1 < symbols.Count; i++)
                if (_merges.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    best = (symbols[i], symbols[i + 1]);
                }
            if (bestRank == int.MaxValue) break;

            var merged = new List<string>(symbols.Count);
            for (int i = 0; i < symbols.Count;)
            {
                if (i + 1 < symbols.Count && symbols[i] == best.Left && symbols[i + 1] == best.Right)
                {
                    merged.Add(symbols[i] + symbols[i + 1]);
                    i += 2;
                }
                else
                {
                    merged.Add(symbols[i++]);
                }
            }
            symbols = merged;
        }

        foreach (var symbol in symbols)
            output.Add(_vocabulary.TryGetValue(symbol, out int id) ? id : _unknown);
    }

    private static List<string> CodePoints(string text)
    {
        var result = new List<string>();
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                result.Add(char.ConvertFromUtf32(char.ConvertToUtf32(text[i], text[++i])));
            else
                result.Add(text[i].ToString(CultureInfo.InvariantCulture));
        }
        return result;
    }
}
