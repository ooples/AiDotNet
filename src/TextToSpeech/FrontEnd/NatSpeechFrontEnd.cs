using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// NATSpeech's text front end (the authors' FastSpeech, PortaSpeech and SyntaSpeech code base, which Mega-TTS follows):
/// the English processor (<c>data_gen/tts/txt_processors/en.py</c>) that cleans the text and reads it with g2p_en, the
/// base processor's post-processing (silent words stripped from the ends, "|" between words, <c>&lt;BOS&gt;</c> and
/// <c>&lt;EOS&gt;</c>), and <c>TokenTextEncoder</c>'s ids over a phone set.
/// </summary>
internal static class NatSpeechFrontEnd
{
    /// <summary>NATSpeech's punctuation (<c>PUNCS</c>).</summary>
    public const string Punctuation = "!,.?;:";

    /// <summary><c>TokenTextEncoder</c>'s reserved tokens, ids 0–2.</summary>
    public static readonly string[] Reserved = { "<pad>", "<EOS>", "<UNK>" };

    /// <summary>Every phone the English processor writes: g2p_en's ARPAbet with stress, the punctuation, "|" and
    /// <c>&lt;BOS&gt;</c> (<c>&lt;EOS&gt;</c> is reserved).</summary>
    public static string[] EnglishPhones()
    {
        var arpabet = new[]
        {
            "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH",
            "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
            "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW",
            "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
        };
        return arpabet.Concat(Punctuation.Select(c => c.ToString())).Concat(new[] { "|", "<BOS>" }).ToArray();
    }

    /// <summary><c>TokenTextEncoder(vocab_list=phones)</c>'s vocabulary: the reserved tokens, then the phone set sorted
    /// (ordinally, as Python sorts it) without them.</summary>
    public static string[] Vocabulary(IEnumerable<string> phones) =>
        Reserved.Concat(phones.Where(p => !Reserved.Contains(p)).Distinct().OrderBy(p => p, StringComparer.Ordinal)).ToArray();

    /// <summary><c>is_sil_phoneme</c>: empty, or not starting with a letter.</summary>
    public static bool IsSilence(string phone) => phone.Length == 0 || !char.IsLetter(phone[0]);

    private static readonly Regex Quotes = new("['\"()]+", RegexOptions.CultureInvariant);
    private static readonly Regex Dashes = new("[-]+", RegexOptions.CultureInvariant);
    private static readonly Regex NotKept = new("[^ a-z" + Regex.Escape(Punctuation) + "]", RegexOptions.CultureInvariant);
    private static readonly Regex SpacedPunctuation = new(" ?([" + Regex.Escape(Punctuation) + "]) ?", RegexOptions.CultureInvariant);
    private static readonly Regex RepeatedPunctuation = new("([" + Regex.Escape(Punctuation) + "])+", RegexOptions.CultureInvariant);
    private static readonly Regex AnyPunctuation = new("([" + Regex.Escape(Punctuation) + "])", RegexOptions.CultureInvariant);
    private static readonly Regex Spaces = new(@"[ \t\n\r\f\v\u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+", RegexOptions.CultureInvariant);

    /// <summary>The English processor's <c>preprocess_text</c>: numbers in words, accents stripped, lower case, quotes
    /// and brackets dropped, dashes to spaces, only letters, spaces and punctuation kept, repeated punctuation merged and
    /// set apart by spaces.</summary>
    public static string PreprocessEnglish(string text)
    {
        text = G2pEnNumbers.Normalize(text);
        text = new string(text.Normalize(NormalizationForm.FormD).Where(c => CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark).ToArray());
        text = text.ToLowerInvariant();
        text = Quotes.Replace(text, "");
        text = Dashes.Replace(text, " ");
        text = NotKept.Replace(text, "");
        text = SpacedPunctuation.Replace(text, "$1");
        text = RepeatedPunctuation.Replace(text, "$1");
        text = text.Replace("i.e.", "that is");
        text = text.Replace("etc.", "etc");
        text = AnyPunctuation.Replace(text, " $1 ");
        return Spaces.Replace(text, " ");
    }

    /// <summary>
    /// The English processor's <c>process</c> and <c>postprocess</c> (with word separators and <c>&lt;BOS&gt;</c>/<c>&lt;EOS&gt;</c>):
    /// each word with its phones. The cleaned text holds only lower-case words and punctuation set apart by single spaces,
    /// so the g2p's tweet tokenizer splits it at the spaces.
    /// </summary>
    public static List<(string Word, List<string> Phones)> EnglishWords(string text)
    {
        string clean = OuteTTSPromptsStrip(PreprocessEnglish(text));
        var words = clean.Split(' ');
        var structure = words.Select(w => (Word: w, Phones: new List<string>())).ToList();
        if (clean.Length > 0)
        {
            int index = 0;
            foreach (var p in G2pEn.Default.Words(words))
            {
                if (p == " ") index++;
                else structure[index].Phones.Add(p);
            }
        }
        while (structure.Count > 0 && IsSilence(structure[0].Word)) structure.RemoveAt(0);
        while (structure.Count > 0 && IsSilence(structure[^1].Word)) structure.RemoveAt(structure.Count - 1);
        var separated = new List<(string Word, List<string> Phones)>();
        for (int i = 0; i < structure.Count; i++)
        {
            separated.Add(structure[i]);
            if (i != structure.Count - 1 && !IsSilence(structure[i].Word) && !IsSilence(structure[i + 1].Word))
                separated.Add(("|", new List<string> { "|" }));
        }
        separated.Insert(0, ("<BOS>", new List<string> { "<BOS>" }));
        separated.Add(("<EOS>", new List<string> { "<EOS>" }));
        return separated;
    }

    /// <summary>The phones of an English text.</summary>
    public static List<string> EnglishPhonesOf(string text) => EnglishWords(text).SelectMany(w => w.Phones).ToList();

    private const string FullWidth = "：，。！？【】（）％＃＠＆１２３４５６７８９０";
    private const string HalfWidth = ":,.!?[]()%#@&1234567890";
    private static readonly Regex NotKeptChinese = new("[^ A-Za-z\u4e00-\u9fff" + Regex.Escape(Punctuation) + "]", RegexOptions.CultureInvariant);

    /// <summary>The Chinese processor's <c>preprocess_text</c> (the authors' NATSpeech-family code, DiffSinger's
    /// <c>txt_processors/zh.py</c>): full-width punctuation and digits to half width, non-standard words in Chinese
    /// (<see cref="ChineseNswNormalizer"/>), quotes and brackets dropped, dashes to spaces, only letters, Han characters and
    /// punctuation kept, repeated punctuation merged, and every space removed.</summary>
    public static string PreprocessChinese(string text)
    {
        var builder = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            int i = FullWidth.IndexOf(c);
            builder.Append(i >= 0 ? HalfWidth[i] : c);
        }
        text = ChineseNswNormalizer.Normalize(builder.ToString());
        text = Quotes.Replace(text, "");
        text = Dashes.Replace(text, " ");
        text = NotKeptChinese.Replace(text, "");
        text = RepeatedPunctuation.Replace(text, "$1");
        text = AnyPunctuation.Replace(text, " $1 ");
        return Spaces.Replace(text, "");
    }

    /// <summary>
    /// The Chinese processor's <c>process</c> (with tones): "|", then for each pypinyin item its initial and its final with
    /// the tone digit (5 for the neutral tone) and "|", or a non-Han run as itself and "|". Empty initials are dropped,
    /// as encoding the space-joined phones drops them.
    /// </summary>
    public static List<string> ChinesePhonesOf(string text)
    {
        var phones = new List<string> { "|" };
        foreach (var (initial, final) in Pypinyin050.Default.LazyPinyin(PreprocessChinese(text)))
        {
            if (initial == final) phones.Add(initial);
            else
            {
                if (initial.Length > 0) phones.Add(initial);
                phones.Add(final);
            }
            phones.Add("|");
        }
        return phones;
    }

    /// <summary><c>TokenTextEncoder.encode</c> with <c>replace_oov='&lt;UNK&gt;'</c>.</summary>
    public static int[] Encode(IEnumerable<string> phones, IReadOnlyList<string> vocabulary)
    {
        var index = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < vocabulary.Count; i++) index[vocabulary[i]] = i;
        return phones.Select(p => index.TryGetValue(p, out int id) ? id : index["<UNK>"]).ToArray();
    }

    private static string OuteTTSPromptsStrip(string text) => AiDotNet.TextToSpeech.Latest.OuteTTSPrompts.PythonStrip(text);
}
