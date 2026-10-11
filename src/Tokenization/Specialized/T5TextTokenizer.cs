using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.Tokenization.Specialized;

/// <summary>
/// The tokenizer of the pretrained T5 checkpoints (Raffel et al. 2020): t5-base's SentencePiece unigram model, ids for
/// the special strings, and an end-of-sequence id, matching sentencepiece and transformers 4.x's slow
/// <c>T5Tokenizer</c>, so a T5 encoder loaded from those checkpoints reads the ids it was trained on.
/// </summary>
/// <remarks>
/// <para>The vocabulary is the 32,000 SentencePiece pieces (<c>&lt;pad&gt;</c> 0, <c>&lt;/s&gt;</c> 1, <c>&lt;unk&gt;</c> 2)
/// and 100 sentinels, <c>&lt;extra_id_N&gt;</c> = 32099 − N. Special strings in the text map to their ids, the text
/// between them is segmented on its own, and <c>&lt;/s&gt;</c> is appended unless the ids already end with it.</para>
/// </remarks>
public sealed class T5TextTokenizer
{
    private static readonly Lazy<SentencePieceUnigramModel> T5Base = new(() =>
    {
        const string name = "AiDotNet.Tokenization.t5_spiece.model.gz";
        using var stream = typeof(T5TextTokenizer).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidDataException($"The embedded resource {name} is missing.");
        return SentencePieceUnigramModel.FromGzip(stream);
    });

    private readonly SentencePieceUnigramModel _model;

    /// <summary>The tokenizer of t5-base (shared by t5-small, t5-large and the T5 v1.0 checkpoints).</summary>
    public T5TextTokenizer()
        : this(T5Base.Value)
    {
    }

    /// <summary>A T5 tokenizer over another SentencePiece model with T5's special ids.</summary>
    public T5TextTokenizer(SentencePieceUnigramModel model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>The padding id (0).</summary>
    public const int PadId = 0;

    /// <summary>The end-of-sequence id (1).</summary>
    public const int EosId = 1;

    /// <summary>The number of sentinel ids after the SentencePiece pieces.</summary>
    public const int ExtraIds = 100;

    /// <summary>The vocabulary size, pieces plus sentinels (32,100 for t5-base).</summary>
    public int VocabularySize => _model.Count + ExtraIds;

    /// <summary>The ids of <paramref name="text"/>, ending with <c>&lt;/s&gt;</c>.</summary>
    public int[] Encode(string text)
    {
        Guard.NotNull(text);
        var ids = new List<int>();
        int position = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] != '<' || !TrySpecial(text, i, out int id, out int length)) continue;
            ids.AddRange(_model.Encode(text.Substring(position, i - position)));
            ids.Add(id);
            position = i + length;
            i = position - 1;
        }
        ids.AddRange(_model.Encode(text.Substring(position)));
        if (ids.Count == 0 || ids[^1] != EosId) ids.Add(EosId);
        return ids.ToArray();
    }

    // The id and length of the special string at `start` ("</s>", "<pad>", "<unk>" or "<extra_id_N>", N < 100).
    private bool TrySpecial(string text, int start, out int id, out int length)
    {
        foreach (var (literal, value) in new[] { ("</s>", EosId), ("<pad>", PadId), ("<unk>", _model.UnknownId) })
        {
            if (string.CompareOrdinal(text, start, literal, 0, literal.Length) == 0)
            {
                id = value;
                length = literal.Length;
                return true;
            }
        }
        const string prefix = "<extra_id_";
        id = length = 0;
        if (string.CompareOrdinal(text, start, prefix, 0, prefix.Length) != 0) return false;
        int digits = start + prefix.Length, end = digits;
        while (end < text.Length && end - digits < 2 && text[end] >= '0' && text[end] <= '9') end++;
        if (end == digits || end >= text.Length || text[end] != '>') return false;
        int n = int.Parse(text.Substring(digits, end - digits), System.Globalization.CultureInfo.InvariantCulture);
        id = _model.Count + ExtraIds - 1 - n;
        length = end + 1 - start;
        return true;
    }
}
