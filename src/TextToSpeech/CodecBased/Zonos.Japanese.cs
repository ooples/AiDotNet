using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Zonos<T>
{
    private static readonly Regex Digits = new(@"\d+", RegexOptions.CultureInvariant);
    private static readonly object SudachiGate = new();
    private static readonly Dictionary<string, SudachiTokenizer> Sudachi = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// <c>normalize_jp_text</c>: NFKC, each run of digits as kanji (kanjize's <c>number2kanji</c>), then the readings of
    /// Sudachi's morphemes in split mode A joined by spaces, with the full dictionary (<see cref="ZonosOptions.SudachiDictionary"/>).
    /// </summary>
    internal string JapaneseReadings(string text)
    {
        string normalized = text.Normalize(NormalizationForm.FormKC);
        normalized = Digits.Replace(normalized, m => Kanjize(ParseDigits(m.Value)));
        return string.Join(" ", SudachiTokenizerFor().Readings(normalized));
    }

    private SudachiTokenizer SudachiTokenizerFor()
    {
        string dictionary = _options.SudachiDictionary ?? Environment.GetEnvironmentVariable("SUDACHI_DICT")
            ?? throw new InvalidOperationException("Zonos reads Japanese through Sudachi: set ZonosOptions.SudachiDictionary (or SUDACHI_DICT) to "
                + "sudachidict_full's system.dic and ZonosOptions.SudachiResources (or SUDACHI_RESOURCES) to SudachiPy's resources directory.");
        string resources = _options.SudachiResources ?? Environment.GetEnvironmentVariable("SUDACHI_RESOURCES")
            ?? throw new InvalidOperationException("Set ZonosOptions.SudachiResources (or SUDACHI_RESOURCES) to SudachiPy's resources directory "
                + "(char.def, unk.def, rewrite.def).");
        string key = Path.GetFullPath(dictionary) + "|" + Path.GetFullPath(resources);
        lock (SudachiGate)
        {
            if (!Sudachi.TryGetValue(key, out var tokenizer))
                Sudachi[key] = tokenizer = new SudachiTokenizer(SudachiDictionary.Open(dictionary, resources));
            return tokenizer;
        }
    }

    // int() of a run of Unicode decimal digits.
    private static BigInteger ParseDigits(string digits)
    {
        BigInteger value = 0;
        foreach (char c in digits) value = value * 10 + (int)char.GetNumericValue(c);
        return value;
    }

    private static readonly string[] KanjiDigits = { "", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
    private static readonly string[] LargeUnits =
    {
        "", "万", "億", "兆", "京", "垓", "\U00025771", "穣", "溝", "澗", "正", "載", "極", "恒河沙", "阿僧祇", "那由多", "不可思議", "無量大数",
    };

    /// <summary>kanjize 1.6's <c>number2kanji</c> with its default style (all kanji): 0 as 零, each group of four digits as
    /// 千百十 and digits (a 1 before a unit dropped) followed by its large unit.</summary>
    internal static string Kanjize(BigInteger number)
    {
        if (number.IsZero) return "零";
        bool negative = number.Sign < 0;
        number = BigInteger.Abs(number);
        int groups = (number.ToString(CultureInfo.InvariantCulture).Length + 3) / 4;
        if (groups >= LargeUnits.Length) throw new ArgumentOutOfRangeException(nameof(number), "kanjize names numbers below 10^72.");
        var result = new StringBuilder();
        for (int i = groups; i >= 0; i--)
        {
            int group = (int)(number / BigInteger.Pow(10000, i) % 10000);
            if (group == 0) continue;
            int thousands = group / 1000, hundreds = group / 100 % 10, tens = group / 10 % 10, ones = group % 10;
            var part = new StringBuilder();
            if (thousands > 0) part.Append(thousands > 1 ? KanjiDigits[thousands] : "").Append('千');
            if (hundreds > 0) part.Append(hundreds > 1 ? KanjiDigits[hundreds] : "").Append('百');
            if (tens > 0) part.Append(tens > 1 ? KanjiDigits[tens] : "").Append('十');
            if (ones > 0) part.Append(KanjiDigits[ones]);
            result.Append(part).Append(LargeUnits[i]);
        }
        return negative ? "-" + result : result.ToString();
    }
}
