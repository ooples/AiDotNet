using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// The English front end NaturalSpeech 2 and 3 share (Amphion's <c>text/g2p.py</c> and phone set): ARPAbet phones from a
/// lexicon or g2p_en, punctuation as <c>sp</c>.
/// </summary>
internal static class NaturalSpeechArpabet
{
    /// <summary>CMUdict's 84 ARPAbet symbols, then sp, spn, sil, &lt;s&gt; and &lt;/s&gt;.</summary>
    public static readonly IReadOnlyList<string> Symbols = new[]
    {
        "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2", "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2",
        "AY", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2", "F", "G",
        "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2", "OY", "OY0",
        "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
        "sp", "spn", "sil", "<s>", "</s>",
    };

    private static readonly Dictionary<string, int> SymbolIds = Symbols.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i);
    private static readonly ConcurrentDictionary<string, Dictionary<string, string[]>> Lexicons = new(StringComparer.Ordinal);

    private static readonly Regex Separators = new(@"([,;.\-\?\!\s+])", RegexOptions.CultureInvariant);
    private static readonly Regex Punctuation = new(@"\{[^\w\s]?\}", RegexOptions.CultureInvariant);
    private const string TrailingPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    /// <summary>
    /// The text's phone ids (<c>preprocess_english</c>): trailing punctuation stripped, the text split at punctuation and
    /// spaces, each piece looked up in the lexicon or pronounced by g2p_en, a punctuation mark becoming <c>sp</c>.
    /// </summary>
    public static int[] Phonemes(string text, string? lexiconPath)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var lexicon = lexiconPath is null ? null : Lexicons.GetOrAdd(lexiconPath, Read);
        var phones = new List<string>();
        foreach (string w in Separators.Split(text.TrimEnd(TrailingPunctuation.ToCharArray())))
        {
            if (lexicon is not null && lexicon.TryGetValue(w.ToLowerInvariant(), out var known)) phones.AddRange(known);
            else if (w.Length > 0) phones.AddRange(G2pEn.Default.Word(w).Where(p => p != " "));
        }
        string joined = "{" + string.Join("}{", phones) + "}";
        joined = Punctuation.Replace(joined, "{sp}");
        var ids = new List<int>();
        foreach (string phone in joined.Trim('{', '}').Split(new[] { "}{" }, StringSplitOptions.None))
            if (SymbolIds.TryGetValue(phone, out int id)) ids.Add(id);
        if (ids.Count == 0) throw new ArgumentException("The text has no pronounceable phones.", nameof(text));
        return ids.ToArray();
    }

    // read_lexicon: the first pronunciation of each lower-cased word.
    private static Dictionary<string, string[]> Read(string path)
    {
        var lexicon = new Dictionary<string, string[]>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            var parts = Regex.Split(line.TrimEnd('\n'), @"\s+");
            string word = parts[0].ToLowerInvariant();
            if (!lexicon.ContainsKey(word)) lexicon[word] = parts.Skip(1).ToArray();
        }
        return lexicon;
    }
}
