using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// GLaMM's tokenizer: Vicuna's SentencePiece BPE model with GLaMM's added tokens, as the slow LLaMA tokenizer of the
/// transformers revision GLaMM pins (cae78c46) applies them.
/// </summary>
/// <remarks>
/// <para>
/// The added tokens follow the vocabulary in GLaMM's order (<c>&lt;im_start&gt;</c>, <c>&lt;im_end&gt;</c>,
/// <c>&lt;bbox&gt;</c>, <c>&lt;point&gt;</c>, <c>[SEG]</c>, <c>&lt;p&gt;</c>, <c>&lt;/p&gt;</c>). Text is split on them and on
/// <c>&lt;s&gt;</c>, <c>&lt;/s&gt;</c> and <c>&lt;unk&gt;</c>; whitespace next to an added token is stripped (that revision's
/// default for tokens added as strings), and each remaining piece of text is encoded by SentencePiece on its own, so
/// each gets the dummy-prefix "▁". Encoding prepends <c>&lt;s&gt;</c>. Padding is <c>&lt;unk&gt;</c>.
/// </para>
/// <para>
/// <see cref="EncodeWithImage"/> is LLaVA's <c>tokenizer_image_token</c>: the prompt split on <c>&lt;image&gt;</c>, each
/// part encoded with its <c>&lt;s&gt;</c>, and the parts joined by the image placeholder (-200), keeping only the first
/// <c>&lt;s&gt;</c>.
/// </para>
/// </remarks>
internal sealed class GlammTokenizer
{
    /// <summary>GLaMM's added tokens, in the order <c>train.py</c> adds them.</summary>
    public static readonly string[] AddedTokens = { "<im_start>", "<im_end>", "<bbox>", "<point>", "[SEG]", "<p>", "</p>" };

    /// <summary>LLaVA's image placeholder id.</summary>
    public const int ImageTokenIndex = -200;

    /// <summary>The label that is not scored.</summary>
    public const int IgnoreIndex = -100;

    private readonly SentencePieceModel _model;
    private readonly string[] _special;

    public GlammTokenizer(SentencePieceModel model)
    {
        if (!model.IsBpe) throw new ArgumentException("GLaMM reads Vicuna's BPE SentencePiece model.", nameof(model));
        _model = model;
        BosId = model.BosId;
        EosId = model.EosId;
        _special = new[] { model.IdToPiece(model.UnknownId), model.IdToPiece(model.BosId), model.IdToPiece(model.EosId) };
    }

    /// <summary>The SentencePiece vocabulary plus the added tokens.</summary>
    public int VocabularySize => _model.Count + AddedTokens.Length;

    public int BosId { get; }

    public int EosId { get; }

    /// <summary>The padding id (the unknown id, as <c>tokenizer.pad_token = tokenizer.unk_token</c>).</summary>
    public int PadId => _model.UnknownId;

    public int IdOf(string addedToken)
    {
        int index = Array.IndexOf(AddedTokens, addedToken);
        if (index < 0) throw new ArgumentException($"{addedToken} is not an added token.", nameof(addedToken));
        return _model.Count + index;
    }

    public int SegId => IdOf("[SEG]");

    public int BboxId => IdOf("<bbox>");

    /// <summary>The llava_v1 system prompt GLaMM's conversations start with.</summary>
    public const string SystemPrompt = "A chat between a curious human and an artificial intelligence assistant. "
        + "The assistant gives helpful, detailed, and polite answers to the human's questions.";

    /// <summary>
    /// The llava_v1 conversation (SeparatorStyle.TWO): the system prompt, then per turn " USER: question ASSISTANT: answer"
    /// closed by &lt;/s&gt;; a turn without an answer ends with "ASSISTANT:" for generation.
    /// </summary>
    public static string Conversation(IReadOnlyList<(string User, string? Assistant)> turns)
    {
        var sb = new System.Text.StringBuilder(SystemPrompt).Append(' ');
        foreach (var (user, assistant) in turns)
        {
            sb.Append("USER: ").Append(user).Append(' ');
            if (assistant is null) sb.Append("ASSISTANT:");
            else sb.Append("ASSISTANT: ").Append(assistant).Append("</s>");
        }
        return sb.ToString();
    }

    /// <summary>
    /// LLaVA's label masking for a llava_v1 conversation (GLaMM's <c>_process_conversation</c>): the labels are the ids with
    /// &lt;s&gt; and every turn's instruction (through "ASSISTANT: ") set to -100, by the official token-length arithmetic.
    /// </summary>
    public int[] MaskInstructions(string conversation, IReadOnlyList<int> ids)
    {
        const string sep = " ASSISTANT: ", sep2 = "</s>";
        bool image = conversation.Contains("<image>");
        int Length(string text) => image ? EncodeWithImage(text).Count : Encode(text).Count;
        var labels = ids.ToArray();
        int cur = 1;
        labels[0] = IgnoreIndex;
        foreach (var round in conversation.Split(new[] { sep2 }, StringSplitOptions.None))
        {
            if (round.Length == 0) break;
            var parts = round.Split(new[] { sep }, StringSplitOptions.None);
            if (parts.Length != 2) throw new ArgumentException($"A turn must hold exactly one \"{sep.Trim()}\": {round}", nameof(conversation));
            int roundLength = Length(round), instructionLength = Length(parts[0] + sep) - 2;
            for (int i = cur; i < Math.Min(labels.Length, cur + instructionLength); i++) labels[i] = IgnoreIndex;
            cur += roundLength;
        }
        for (int i = cur; i < labels.Length; i++) labels[i] = IgnoreIndex;
        return labels;
    }

    /// <summary>The ids of <paramref name="text"/>, with <c>&lt;s&gt;</c> first when <paramref name="addBos"/>.</summary>
    public List<int> Encode(string text, bool addBos = true)
    {
        var ids = new List<int>();
        if (addBos) ids.Add(BosId);
        var parts = Split(text);
        for (int i = 0; i < parts.Count; i++)
        {
            var (part, isToken, added) = parts[i];
            if (!isToken) continue;
            if (added)
            {
                if (i > 0 && !parts[i - 1].IsToken) parts[i - 1] = (RStrip(parts[i - 1].Text), false, false);
                if (i + 1 < parts.Count && !parts[i + 1].IsToken) parts[i + 1] = (LStrip(parts[i + 1].Text), false, false);
            }
        }
        foreach (var (part, isToken, added) in parts)
        {
            if (part.Length == 0) continue;
            if (!isToken) ids.AddRange(_model.Encode(part));
            else if (added) ids.Add(IdOf(part));
            else ids.Add(_model.PieceToId(part));
        }
        return ids;
    }

    /// <summary>LLaVA's <c>tokenizer_image_token</c>.</summary>
    public List<int> EncodeWithImage(string prompt)
    {
        var chunks = prompt.Split(new[] { "<image>" }, StringSplitOptions.None).Select(c => Encode(c)).ToList();
        var ids = new List<int>();
        int offset = 0;
        if (chunks.Count > 0 && chunks[0].Count > 0 && chunks[0][0] == BosId)
        {
            offset = 1;
            ids.Add(BosId);
        }
        for (int i = 0; i < chunks.Count; i++)
        {
            if (i > 0) ids.Add(ImageTokenIndex);   // insert_separator's [-200] * (offset + 1), then [offset:]
            ids.AddRange(chunks[i].Skip(offset));
        }
        return ids;
    }

    /// <summary>
    /// That revision's <c>decode(ids, skip_special_tokens=False)</c>: SentencePiece pieces decoded in runs (each run gets a
    /// leading space, the revision's <c>_maybe_add_prefix_space</c>, which compares tokens with a set of ids and so always
    /// adds it), added tokens kept, everything joined by spaces, and the tokenization clean-up applied.
    /// </summary>
    public string Decode(IEnumerable<int> ids)
    {
        var texts = new List<string>();
        var run = new List<int>();
        void Flush()
        {
            if (run.Count == 0) return;
            texts.Add(PiecesToString(run));
            run.Clear();
        }
        foreach (int id in ids)
        {
            if (id >= _model.Count && id < VocabularySize)
            {
                Flush();
                texts.Add(AddedTokens[id - _model.Count]);
            }
            else if (id >= 0 && id < _model.Count) run.Add(id);
        }
        Flush();
        return Tokenization.TokenizationCleanup.CleanUpSpaces(string.Join(" ", texts));
    }

    // LlamaTokenizer.convert_tokens_to_string: special tokens spliced in, the rest decoded by SentencePiece, and a leading space.
    private string PiecesToString(List<int> ids)
    {
        var output = new System.Text.StringBuilder();
        var current = new List<int>();
        bool previousSpecial = false;
        foreach (int id in ids)
        {
            string piece = _model.IdToPiece(id);
            if (Array.IndexOf(_special, piece) >= 0)
            {
                if (!previousSpecial) output.Append(' ');
                output.Append(_model.Decode(current)).Append(piece);
                previousSpecial = true;
                current.Clear();
            }
            else
            {
                current.Add(id);
                previousSpecial = false;
            }
        }
        output.Append(_model.Decode(current));
        return " " + output;
    }

    // The text split at every added or special token (leftmost, longest first): (text, is a token, is an added token).
    private List<(string Text, bool IsToken, bool Added)> Split(string text)
    {
        var parts = new List<(string, bool, bool)>();
        int start = 0, i = 0;
        while (i < text.Length)
        {
            string? match = null;
            bool added = false;
            foreach (var token in AddedTokens)
                if (string.CompareOrdinal(text, i, token, 0, token.Length) == 0 && (match is null || token.Length > match.Length)) { match = token; added = true; }
            foreach (var token in _special)
                if (string.CompareOrdinal(text, i, token, 0, token.Length) == 0 && (match is null || token.Length > match.Length)) { match = token; added = false; }
            if (match is null)
            {
                i++;
                continue;
            }
            if (i > start) parts.Add((text.Substring(start, i - start), false, false));
            parts.Add((match, true, added));
            i += match.Length;
            start = i;
        }
        if (start < text.Length) parts.Add((text.Substring(start), false, false));
        return parts;
    }

    // Python's str.isspace: Unicode whitespace (bidirectional WS, B, S, or category Zs), which includes U+001C to U+001F.
    private static bool IsPythonSpace(char c)
        => c is >= '\t' and <= '\r' or >= '\x1c' and <= ' ' or '\x85' or '\xa0' or '\u1680' or >= '\u2000' and <= '\u200a'
            or '\u2028' or '\u2029' or '\u202f' or '\u205f' or '\u3000';

    private static string LStrip(string s)
    {
        int i = 0;
        while (i < s.Length && IsPythonSpace(s[i])) i++;
        return s.Substring(i);
    }

    private static string RStrip(string s)
    {
        int i = s.Length;
        while (i > 0 && IsPythonSpace(s[i - 1])) i--;
        return s.Substring(0, i);
    }
}
