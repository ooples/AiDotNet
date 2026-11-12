using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// MeloTTS's mixed Chinese-English front end (myshell-ai/MeloTTS <c>melo/text/chinese_mix.py</c>, its default
/// <c>_g2p_v2</c>): normalized as the Chinese front end does but keeping Latin letters and whitespace (collapsed to one
/// space), then each sentence split into runs of Latin letters and whitespace, read by the English front end from
/// multilingual BERT's word pieces (tones moved past the Chinese and Japanese ones), and the Chinese between them, read by
/// the Chinese front end.
/// </summary>
internal sealed class MeloChineseMixFrontEnd
{
    private static readonly Lazy<MeloChineseMixFrontEnd> SharedInstance = new(() => new MeloChineseMixFrontEnd(), isThreadSafe: true);
    private readonly WordPieceTokenizer _tokenizer = WordPieceTokenizer.BertBaseMultilingualUncased();

    public static MeloChineseMixFrontEnd Default => SharedInstance.Value;

    private const string Punctuation = "!?…,.'-";
    // Python's \s (str.isspace): .NET's \s plus the separators U+001C to U+001F.
    private const string Whitespace = @"\s\x1c-\x1f";
    private static readonly Regex Outside = new("[^一-龥_a-zA-Z" + Whitespace + Regex.Escape(Punctuation) + "]+", RegexOptions.CultureInvariant);
    private static readonly Regex Spaces = new("[" + Whitespace + "]+", RegexOptions.CultureInvariant);
    private static readonly Regex SentenceSplit = new("(?<=[" + Regex.Escape(Punctuation) + "])[" + Whitespace + "]*", RegexOptions.CultureInvariant);
    private static readonly Regex LatinRuns = new("[a-zA-Z" + Whitespace + "]+", RegexOptions.CultureInvariant);

    /// <summary>The word pieces multilingual BERT cuts text into (the pieces the phones are counted against).</summary>
    public List<string> WordPieces(string text) => _tokenizer.Tokenize(text);

    /// <summary><c>chinese_mix.text_normalize</c>.</summary>
    public string Normalize(string text) =>
        Spaces.Replace(Outside.Replace(MeloChineseFrontEnd.NumeralsAndPunctuation(text), ""), " ");

    /// <summary>
    /// <c>chinese_mix.g2p(text)</c> (<c>impl='v2'</c>) of normalized text: the phones, their tones (Chinese 0–5, English
    /// past the Chinese and Japanese tones), and the phones per Chinese character or English word piece, with the
    /// silence symbol at each end.
    /// </summary>
    public (List<string> Phones, List<int> Tones, List<int> PhonesPerUnit) G2P(string text)
    {
        var phones = new List<string>();
        var tones = new List<int>();
        var perUnit = new List<int>();
        int englishTones = AiDotNet.TextToSpeech.EndToEnd.MeloSymbols.ToneStart("EN");
        foreach (var sentence in SentenceSplit.Split(text).Where(s => !IsBlank(s)))
        {
            int at = 0;
            foreach (Match run in LatinRuns.Matches(sentence))
            {
                if (run.Index > at) AppendChinese(sentence.Substring(at, run.Index - at), phones, tones, perUnit);
                var (enPhones, enTones, enPerPiece) = MeloEnglishFrontEnd.Default.G2P(WordPieces(run.Value), padStartEnd: false);
                phones.AddRange(enPhones);
                tones.AddRange(enTones.Select(t => t + englishTones));
                perUnit.AddRange(enPerPiece);
                at = run.Index + run.Length;
            }
            if (at < sentence.Length) AppendChinese(sentence.Substring(at), phones, tones, perUnit);
        }
        if (perUnit.Sum() != phones.Count)
            throw new ArgumentException($"The phones of '{text}' do not add up to the phones per unit.", nameof(text));
        phones.Insert(0, "_");
        phones.Add("_");
        tones.Insert(0, 0);
        tones.Add(0);
        perUnit.Insert(0, 1);
        perUnit.Add(1);
        return (phones, tones, perUnit);
    }

    private static bool IsBlank(string s) => Spaces.Replace(s, "").Length == 0;

    private static void AppendChinese(string text, List<string> phones, List<int> tones, List<int> perUnit) =>
        MeloChineseFrontEnd.Default.AppendSegment(text, phones, tones, perUnit);
}
