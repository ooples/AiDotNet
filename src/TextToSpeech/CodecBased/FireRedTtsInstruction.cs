using System.Text;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// The paralinguistic markup of FireRedTTS's instruction tuning (§4.2, Table 1): token-insertion behaviours are tags in
/// the text (<c>[hic]</c>, <c>[rep]</c>, <c>[elong]</c>, <c>[sss]</c>, <c>[tsk]</c>, <c>[breath]</c>, <c>[laugh]</c>), which
/// the tokenizer reads as tokens; embedding-injection behaviours mark the Chinese character before them (<c>ˆ</c> speaking
/// with a laugh, <c>@</c> emphasis, <c>P</c> filled pause, <c>C</c> confirmation, <c>R</c> realization, <c>S</c>
/// surprise), and the label's embedding is added to that character's tokens.
/// </summary>
/// <remarks>The paper does not say how labels follow the text through normalization; here a marked character is found
/// again, in order, in the normalized text, and every token whose bytes overlap it carries the label.</remarks>
internal static class FireRedTtsInstruction
{
    /// <summary>The label table's rows: none, then the six embedding-injection labels.</summary>
    public const int LabelCount = 7;

    private static int LabelOf(char marker) => marker switch
    {
        'ˆ' or '^' => 1,
        '@' => 2,
        'P' => 3,
        'C' => 4,
        'R' => 5,
        'S' => 6,
        _ => 0,
    };

    private static bool IsChinese(char c) => c >= '一' && c <= '龥';

    // A marker counts after a Chinese character (the letters only there, so English capitals stay text).
    private static bool IsMarker(string text, int i) =>
        i > 0 && IsChinese(text[i - 1]) && LabelOf(text[i]) != 0;

    /// <summary>The text without its embedding-injection markers.</summary>
    public static string Strip(string text)
    {
        var builder = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i++)
            if (!IsMarker(text, i)) builder.Append(text[i]);
        return builder.ToString();
    }

    /// <summary>The marked characters in order, with their labels, consumed as the pieces are encoded.</summary>
    public static Queue<(char Character, int Label)> Parse(string text)
    {
        var marked = new Queue<(char, int)>();
        for (int i = 0; i < text.Length; i++)
            if (IsMarker(text, i)) marked.Enqueue((text[i - 1], LabelOf(text[i])));
        return marked;
    }

    /// <summary>A piece's token ids and their labels (the marked characters this piece holds taken from the queue).</summary>
    public static (int[] Ids, int[]? Labels) Encode(FireRedTtsTokenizer tokenizer, string piece, Queue<(char Character, int Label)> marked)
    {
        var (normalized, language) = FireRedTtsTextNormalizer.Normalize(PythonText.Strip(piece));
        string text = "[" + language + "]" + normalized;
        var ids = tokenizer.Encoding.Encode(text);
        var spans = new List<(int Start, int End, int Label)>();
        int from = 2 + language.Length;
        while (marked.Count > 0)
        {
            var (character, label) = marked.Peek();
            int at = text.IndexOf(character, from);
            if (at < 0) break;
            marked.Dequeue();
            int start = Encoding.UTF8.GetByteCount(text.Substring(0, at));
            spans.Add((start, start + Encoding.UTF8.GetByteCount(character.ToString()), label));
            from = at + 1;
        }
        if (spans.Count == 0) return (ids, null);
        var labels = new int[ids.Length];
        int offset = 0;
        for (int i = 0; i < ids.Length; i++)
        {
            int end = offset + tokenizer.Encoding.ByteLength(ids[i]);
            foreach (var (s, e, label) in spans)
                if (s < end && e > offset) labels[i] = label;
            offset = end;
        }
        return (ids, labels);
    }
}
