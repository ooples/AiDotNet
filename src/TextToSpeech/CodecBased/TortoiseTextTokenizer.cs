using System.IO.Compression;
using System.Text.Json;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The released character-BPE vocabulary, whitespace pre-tokenizer, and English cleaner.</summary>
internal sealed class TortoiseTextTokenizer
{
    private readonly Dictionary<string, int> _vocabulary, _merges, _added;
    private readonly Regex _addedPattern;
    private readonly int _unknown;
    internal static readonly Lazy<TortoiseTextTokenizer> Default = new(() =>
    {
        using var stream = typeof(TortoiseTextTokenizer).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.tortoise_tokenizer.json.gz")
            ?? throw new InvalidDataException("Missing released TorToiSe tokenizer.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip);
        return new TortoiseTextTokenizer(reader.ReadToEnd());
    });

    internal TortoiseTextTokenizer(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement; var model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE" || root.GetProperty("pre_tokenizer").GetProperty("type").GetString() != "Whitespace")
            throw new NotSupportedException("TorToiSe requires its character-BPE model with Whitespace pre-tokenization.");
        _vocabulary = model.GetProperty("vocab").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal);
        _merges = model.GetProperty("merges").EnumerateArray().Select((value, index) => (Pair: value.GetString()!, Rank: index))
            .ToDictionary(p => p.Pair, p => p.Rank, StringComparer.Ordinal);
        _added = root.GetProperty("added_tokens").EnumerateArray().ToDictionary(p => p.GetProperty("content").GetString()!, p => p.GetProperty("id").GetInt32(), StringComparer.Ordinal);
        _addedPattern = new Regex("(" + string.Join("|", _added.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape)) + ")",
            RegexOptions.CultureInvariant, TimeSpan.FromSeconds(5));
        _unknown = _vocabulary[model.GetProperty("unk_token").GetString()!];
    }

    internal int VocabularySize => _vocabulary.Values.Max() + 1;

    private static readonly (string Short, string Long)[] Abbreviations =
    [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"),
        ("jr", "junior"), ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"),
        ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
        ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort")
    ];

    internal static string Clean(string text)
    {
        text = G2pEnNumbers.Normalize(AsciiTransliterator.Convert(text).ToLowerInvariant());
        foreach (var (shortName, longName) in Abbreviations)
            text = Regex.Replace(text, @"\b" + shortName + @"\.", longName, RegexOptions.CultureInvariant);
        return Regex.Replace(text, @"\s+", " ", RegexOptions.CultureInvariant).Replace("\"", "");
    }

    internal int[] Encode(string text)
    {
        var result = new List<int>();
        foreach (string part in _addedPattern.Split(Clean(text).Replace(" ", "[SPACE]")))
        {
            if (_added.TryGetValue(part, out int added)) { result.Add(added); continue; }
            foreach (Match match in Regex.Matches(part, @"\w+|[^\w\s]+", RegexOptions.CultureInvariant))
            {
                var symbols = match.Value.Select(c => c.ToString()).ToList();
                while (symbols.Count > 1)
                {
                    int best = -1, rank = int.MaxValue;
                    for (int i = 0; i + 1 < symbols.Count; i++)
                        if (_merges.TryGetValue(symbols[i] + " " + symbols[i + 1], out int candidate) && candidate < rank)
                        { best = i; rank = candidate; }
                    if (best < 0) break;
                    symbols[best] += symbols[best + 1]; symbols.RemoveAt(best + 1);
                }
                foreach (string symbol in symbols) result.Add(_vocabulary.TryGetValue(symbol, out int id) ? id : _unknown);
            }
        }
        return result.ToArray();
    }
}
