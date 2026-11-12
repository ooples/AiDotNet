namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>The character BPE in Plachtaa/VALL-E-X utils/g2p/bpe_69.json (70 rows, no merges).</summary>
internal static class VallEXCommunityPhoneTokenizer
{
    internal static readonly string[] Vocabulary =
    [
        "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "!", "#", "*", ",", "-", ".", "=", "?", "N", "Q", "^", "_", "`",
        "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "s", "t", "u", "v", "w", "x", "y", "z", "~",
        "æ", "ç", "ð", "ŋ", "ɑ", "ɔ", "ə", "ɛ", "ɥ", "ɪ", "ɫ", "ɯ", "ɸ", "ɹ", "ɾ", "ʃ", "ʊ", "ʑ", "ʒ", "ʰ", "ˈ", "ˌ", "θ", "…", "⁼", "↑", "→", "↓",
    ];
    private static readonly Dictionary<string, int> Ids = Vocabulary.Select((value, index) => (value, index))
        .ToDictionary(pair => pair.value, pair => pair.index, StringComparer.Ordinal);

    internal static int[] Encode(IEnumerable<string> symbols)
    {
        string phones = string.Concat(symbols).Replace(' ', '_');
        var result = new List<int>();
        for (int i = 0; i < phones.Length;)
        {
            int special = -1;
            if (phones[i] == '[')
                for (int token = 0; token < 5; token++)
                    if (i + Vocabulary[token].Length <= phones.Length &&
                        string.CompareOrdinal(phones, i, Vocabulary[token], 0, Vocabulary[token].Length) == 0)
                    { special = token; break; }
            if (special >= 0) { result.Add(special); i += Vocabulary[special].Length; continue; }
            int width = char.IsHighSurrogate(phones[i]) && i + 1 < phones.Length && char.IsLowSurrogate(phones[i + 1]) ? 2 : 1;
            if (!char.IsWhiteSpace(phones[i])) result.Add(Ids.TryGetValue(phones.Substring(i, width), out int id) ? id : 0);
            i += width;
        }
        return result.ToArray();
    }
}
