using System.Numerics;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's English cardinal/ordinal rules; distinct from inflect's number reader.</summary>
internal sealed class Num2WordsEnglish : Num2WordsBase
{
    private static readonly Lazy<JObject> Data = new(() => JObject.Parse(string.Join("\n", ChineseText.ReadLines("num2words_english.json.gz"))));
    private static readonly Lazy<Num2WordsEnglish> Shared = new(() => new Num2WordsEnglish());
    internal static Num2WordsEnglish Default => Shared.Value;

    private Num2WordsEnglish() : base(((JArray)Data.Value["cards"]!).Select(entry =>
        (BigInteger.Parse((string)entry[0]!, System.Globalization.CultureInfo.InvariantCulture), (string)entry[1]!)),
        "minus ", "point", " ", "en") { }

    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
    {
        if (l.IsOne && r < 100) return (right, r);
        if (100 > l && l > r) return ($"{left}-{right}", l + r);
        if (l >= 100 && 100 > r) return ($"{left} and {right}", l + r);
        if (r > l) return ($"{left} {right}", l * r);
        return ($"{left}, {right}", l + r);
    }

    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        var words = ToCardinal(value).Split(' ');
        var last = words[words.Length - 1].Split('-');
        string word = last[last.Length - 1];
        string? ordinal = (string?)Data.Value["ordinals"]![word];
        last[last.Length - 1] = ordinal ?? (word.EndsWith("y", StringComparison.Ordinal) ? word.Substring(0, word.Length - 1) + "ieth" : word + "th");
        words[words.Length - 1] = string.Join("-", last);
        return string.Join(" ", words);
    }
}
