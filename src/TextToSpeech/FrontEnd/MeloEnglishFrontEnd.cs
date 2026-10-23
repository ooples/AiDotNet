using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// MeloTTS's English front end (myshell-ai/MeloTTS <c>melo/text/english.py</c>): its text normalization, and its G2P —
/// the text split into bert-base-uncased words, each word looked up in MeloTTS's syllabified CMU dictionary or else
/// converted by g2p_en, stresses turned into tones (unstressed 1, primary 2, secondary 3, others 0), each word's phones
/// spread over its word pieces, and a silence symbol at each end.
/// </summary>
internal sealed class MeloEnglishFrontEnd
{
    private static readonly Lazy<MeloEnglishFrontEnd> Shared = new(() => new MeloEnglishFrontEnd());
    private readonly Dictionary<string, string[][]> _dictionary = new(StringComparer.Ordinal);
    private readonly WordPieceTokenizer _tokenizer = WordPieceTokenizer.BertBaseUncased();

    private static readonly HashSet<string> Arpa = new(StringComparer.Ordinal)
    {
        "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B",
        "CH", "D", "DH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY0",
        "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1",
        "UH2", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
    };

    public static MeloEnglishFrontEnd Default => Shared.Value;

    private MeloEnglishFrontEnd()
    {
        using var stream = typeof(MeloEnglishFrontEnd).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.melo_cmudict.tsv.gz")
            ?? throw new InvalidDataException("The embedded resource melo_cmudict.tsv.gz is missing.");
        using var reader = new StreamReader(new GZipStream(stream, CompressionMode.Decompress), Encoding.UTF8);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            int tab = line.IndexOf('\t');
            if (tab <= 0) continue;
            _dictionary[line.Substring(0, tab)] = line.Substring(tab + 1).Split(new[] { " - " }, StringSplitOptions.None)
                .Select(s => s.Split(' ')).ToArray();
        }
    }

    /// <summary>The bert-base-uncased word pieces of the text, as BERT's tokenizer splits it.</summary>
    public List<string> WordPieces(string text) => _tokenizer.Tokenize(text);

    // ------------------------------------------------------------------ normalization

    private static readonly Regex Time = new(@"\b
                          ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
                          :
                          ([0-5][0-9])                            # minutes
                          \s*(a\\.m\\.|am|pm|p\\.m\\.|a\\.m|p\\.m)? # am/pm
                          \b", RegexOptions.IgnoreCase | RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant);
    private static readonly Regex CommaNumber = new(@"([0-9][0-9\,]+[0-9])", RegexOptions.CultureInvariant);
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Currency = new(@"(£|\$|¥)([0-9\,\.]*[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Ordinal = new(@"[0-9]+(st|nd|rd|th)", RegexOptions.CultureInvariant);
    private static readonly Regex Number = new(@"-?[0-9]+", RegexOptions.CultureInvariant);
    private static readonly (Regex Pattern, string Replacement)[] Abbreviations = new[]
    {
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"), ("jr", "junior"), ("maj", "major"),
        ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"),
        ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    }.Select(p => (new Regex($@"\b{p.Item1}\.", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant), p.Item2)).ToArray();

    /// <summary><c>text_normalize</c>: lower case, times, numbers and currency in words, abbreviations expanded.</summary>
    public string Normalize(string text)
    {
        text = text.ToLowerInvariant();
        text = Time.Replace(text, ExpandTime);
        text = CommaNumber.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = Currency.Replace(text, ExpandCurrency);
        text = DecimalNumber.Replace(text, m => m.Groups[1].Value.Replace(".", " point "));
        text = Ordinal.Replace(text, m => InflectNumberWords.NumberToWords(m.Value));
        text = Number.Replace(text, m => G2pEnNumbers.ExpandNumber(m.Value, zeroOhAboveThousand: true));
        foreach (var (pattern, replacement) in Abbreviations) text = pattern.Replace(text, replacement);
        return text;
    }

    private static string ExpandTime(Match m)
    {
        int hour = int.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        bool pastNoon = hour >= 12;
        if (hour > 12) hour -= 12;
        else if (hour == 0)
        {
            hour = 12;
            pastNoon = true;
        }
        var time = new List<string> { InflectNumberWords.NumberToWords(hour.ToString(System.Globalization.CultureInfo.InvariantCulture)) };
        int minute = int.Parse(m.Groups[6].Value, System.Globalization.CultureInfo.InvariantCulture);
        if (minute > 0)
        {
            if (minute < 10) time.Add("oh");
            time.Add(InflectNumberWords.NumberToWords(minute.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }
        if (!m.Groups[7].Success) time.Add(pastNoon ? "p m" : "a m");
        else time.AddRange(m.Groups[7].Value.Replace(".", "").Select(c => c.ToString()));
        return string.Join(" ", time);
    }

    // number_norm._expand_currency: the amount and its unit in words (the digits are read later).
    private static string ExpandCurrency(Match m)
    {
        string unit = m.Groups[1].Value, value = m.Groups[2].Value;
        // (one, many, one hundredth, many hundredths); ¥ has no singular and no "one sen".
        var (one, many, cent, cents) = unit switch
        {
            "$" => ("dollar", "dollars", "cent", "cents"),
            "£" => ("pound sterling", "pounds sterling", "penny", "pence"),
            _ => ("yen", "yen", "sen", "sen"),
        };
        if (unit == "¥") one = many;
        var parts = value.Replace(",", "").Split('.');
        if (parts.Length > 2) return $"{value} {many}";
        var text = new List<string>();
        var integer = parts[0].Length > 0 ? System.Numerics.BigInteger.Parse(parts[0]) : 0;
        if (integer > 0) text.Add($"{integer} {(integer == 1 ? one : many)}");
        var fraction = parts.Length > 1 && parts[1].Length > 0 ? System.Numerics.BigInteger.Parse(parts[1]) : 0;
        if (fraction > 0) text.Add($"{fraction} {(fraction == 1 && unit != "¥" ? cent : cents)}");
        if (text.Count == 0) return $"zero {many}";
        return string.Join(" ", text);
    }

    // ------------------------------------------------------------------ G2P

    private static (string Phone, int Tone) Refine(string phone)
    {
        int tone = 0;
        if (phone.Length > 0 && char.IsDigit(phone[phone.Length - 1]))
        {
            tone = phone[phone.Length - 1] - '0' + 1;
            phone = phone.Substring(0, phone.Length - 1);
        }
        return (phone.ToLowerInvariant(), tone);
    }

    /// <summary>japanese.distribute_phone: phones dealt one at a time to the word with the fewest so far.</summary>
    public static int[] DistributePhones(int phones, int words)
    {
        var perWord = new int[words];
        for (int task = 0; task < phones; task++)
        {
            int min = perWord.Min(), index = Array.IndexOf(perWord, min);
            perWord[index]++;
        }
        return perWord;
    }

    private static string PostReplace(string phone)
    {
        phone = phone switch
        {
            "：" => ",", "；" => ",", "，" => ",", "。" => ".", "！" => "!", "？" => "?", "\n" => ".", "·" => ",", "、" => ",", "..." => "…",
            "v" => "V", _ => phone,
        };
        return AiDotNet.TextToSpeech.EndToEnd.MeloSymbols.Contains(phone) ? phone : "UNK";
    }

    /// <summary>
    /// <c>english.g2p(text)</c> of normalized text: the phones, their tones (0–3, before the language offset), and the
    /// number of phones of each BERT word piece, with the silence symbol and its piece at each end.
    /// </summary>
    public (List<string> Phones, List<int> Tones, List<int> PhonesPerPiece) G2P(string text)
    {
        var groups = new List<List<string>>();
        foreach (var token in WordPieces(text))
        {
            if (!token.StartsWith("#", StringComparison.Ordinal)) groups.Add(new List<string> { token });
            else if (groups.Count == 0) throw new ArgumentException("The text starts with a word piece continuation.", nameof(text));
            else groups[groups.Count - 1].Add(token.Replace("#", ""));
        }
        var phones = new List<string>();
        var tones = new List<int>();
        var perPiece = new List<int>();
        foreach (var group in groups)
        {
            string word = string.Concat(group);
            int count = 0;
            if (_dictionary.TryGetValue(word.ToUpperInvariant(), out var syllables))
            {
                foreach (var syllable in syllables)
                    foreach (var phone in syllable)
                    {
                        var (p, t) = Refine(phone);
                        phones.Add(p);
                        tones.Add(t);
                        count++;
                    }
            }
            else
            {
                foreach (var phone in G2pEn.Default.Word(word).Where(p => p != " "))
                {
                    if (Arpa.Contains(phone))
                    {
                        var (p, t) = Refine(phone);
                        phones.Add(p);
                        tones.Add(t);
                    }
                    else
                    {
                        phones.Add(phone);
                        tones.Add(0);
                    }
                    count++;
                }
            }
            perPiece.AddRange(DistributePhones(count, group.Count));
        }
        phones = phones.Select(PostReplace).ToList();
        phones.Insert(0, "_");
        phones.Add("_");
        tones.Insert(0, 0);
        tones.Add(0);
        perPiece.Insert(0, 1);
        perPiece.Add(1);
        return (phones, tones, perPiece);
    }
}
