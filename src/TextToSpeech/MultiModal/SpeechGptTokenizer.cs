using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// SpeechGPT's tokenizer as the release runs it (transformers 4.29's slow <c>LlamaTokenizer</c>): LLaMA's SentencePiece
/// model (<c>tokenizer.model</c>, BPE with byte fallback) and the added tokens of <c>added_tokens.json</c> (the 1000
/// unit tokens <c>&lt;0&gt;</c> … <c>&lt;999&gt;</c>, <c>&lt;sosp&gt;</c>, <c>&lt;eosp&gt;</c>, <c>[Human]</c>,
/// <c>[SpeechGPT]</c>, <c>&lt;eoh&gt;</c>, <c>&lt;eos&gt;</c>).
/// </summary>
/// <remarks>
/// <c>tokenize</c> splits the text at the added tokens (the trie's leftmost, longest match), strips the white space on
/// both sides of each (added as plain strings, they strip by default), and encodes every other piece on its own with
/// SentencePiece, the dummy prefix included; <c>&lt;s&gt;</c> (1) begins the ids.
/// </remarks>
internal sealed class SpeechGptTokenizer
{
    private readonly SentencePieceModel _model;
    private readonly Dictionary<string, int> _added;
    private readonly Dictionary<int, string> _addedById;
    private readonly HashSet<string> _special = new(StringComparer.Ordinal) { "<s>", "</s>", "<unk>" };

    public SpeechGptTokenizer(SentencePieceModel model, IReadOnlyDictionary<string, int> added)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _added = (added ?? throw new ArgumentNullException(nameof(added))).ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
        _addedById = _added.ToDictionary(p => p.Value, p => p.Key);
        _byFirst = _added.Keys.Concat(_special).GroupBy(t => t[0]).ToDictionary(g => g.Key, g => g.OrderByDescending(t => t.Length).ToArray());
    }

    private readonly Dictionary<char, string[]> _byFirst;

    /// <summary>Reads <c>tokenizer.model</c> and <c>added_tokens.json</c> from a model directory.</summary>
    public static SpeechGptTokenizer FromDirectory(string directory)
    {
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        var model = new SentencePieceModel(File.ReadAllBytes(Path.Combine(directory, "tokenizer.model")));
        var added = JObject.Parse(File.ReadAllText(Path.Combine(directory, "added_tokens.json"))).Properties()
            .ToDictionary(p => p.Name, p => (int)p.Value, StringComparer.Ordinal);
        return new SpeechGptTokenizer(model, added);
    }

    /// <summary>The ids: the SentencePiece pieces, then the added tokens.</summary>
    public int VocabularySize => Math.Max(_model.Count, _added.Count == 0 ? 0 : _added.Values.Max() + 1);

    public int BosId => _model.BosId;
    public int EosId => _model.EosId;

    /// <summary>The id of an added token such as <c>&lt;sosp&gt;</c> or <c>&lt;12&gt;</c>.</summary>
    public int TokenId(string token) => _added.TryGetValue(token, out int id) ? id : throw new KeyNotFoundException($"'{token}' is not an added token.");

    // The trie's split: at each position the longest token that starts there, scanning left to right.
    private List<(string Text, bool Token)> Split(string text)
    {
        var parts = new List<(string, bool)>();
        int start = 0, i = 0;
        while (i < text.Length)
        {
            string? match = null;
            if (_byFirst.TryGetValue(text[i], out var candidates))
                foreach (var token in candidates)
                    if (string.CompareOrdinal(text, i, token, 0, token.Length) == 0)
                    {
                        match = token;
                        break;
                    }
            if (match is null)
            {
                i++;
                continue;
            }
            if (i > start) parts.Add((text.Substring(start, i - start), false));
            parts.Add((match, true));
            i += match.Length;
            start = i;
        }
        if (start < text.Length) parts.Add((text.Substring(start), false));
        return parts;
    }

    /// <summary>The ids of <paramref name="text"/> (<c>tokenizer(text).input_ids</c>), <c>&lt;s&gt;</c> first unless
    /// <paramref name="addBos"/> is false.</summary>
    public int[] Encode(string text, bool addBos = true)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var parts = Split(text);
        // Added tokens given as strings eat the white space beside them; the special tokens (AddedToken) do not.
        for (int i = 0; i < parts.Count; i++)
        {
            if (!parts[i].Token || _special.Contains(parts[i].Text)) continue;
            if (i + 1 < parts.Count && !parts[i + 1].Token) parts[i + 1] = (parts[i + 1].Text.TrimStart(), false);
            if (i > 0 && !parts[i - 1].Token) parts[i - 1] = (parts[i - 1].Text.TrimEnd(), false);
        }
        var ids = new List<int>();
        if (addBos) ids.Add(BosId);
        foreach (var (piece, token) in parts)
        {
            if (piece.Length == 0) continue;
            if (token) ids.Add(_added.TryGetValue(piece, out int id) ? id : _model.PieceToId(piece));
            else ids.AddRange(_model.Encode(piece));
        }
        return ids.ToArray();
    }

    /// <summary>The text of ids (<c>decode(..., skip_special_tokens=True)</c>): SentencePiece's text between the added
    /// tokens, the added tokens written out, the pieces joined by single spaces.</summary>
    public string Decode(IEnumerable<int> ids)
    {
        var pieces = new List<string>();
        var run = new List<int>();
        void Flush()
        {
            if (run.Count == 0) return;
            pieces.Add(_model.Decode(run));
            run.Clear();
        }
        foreach (int id in ids)
        {
            if (id == BosId || id == EosId || id == _model.UnknownId) continue;
            if (_addedById.TryGetValue(id, out var token))
            {
                Flush();
                pieces.Add(token);
            }
            else
            {
                run.Add(id);
            }
        }
        Flush();
        return string.Join(" ", pieces);
    }
}
