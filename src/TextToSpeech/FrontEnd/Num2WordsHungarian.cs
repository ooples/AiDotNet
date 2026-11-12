using System.Globalization;
using System.Numerics;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's Hungarian composition, contextual két, hyphens and fractional denominators.</summary>
internal sealed class Num2WordsHungarian : Num2WordsEuropean
{
    private static readonly Lazy<Num2WordsHungarian> Shared = new(() => new Num2WordsHungarian());
    internal static Num2WordsHungarian Default => Shared.Value;
    private readonly IReadOnlyDictionary<BigInteger, string> _cards;
    private Num2WordsHungarian() : base("hu") => _cards = ((JArray)LanguageData["cards"]!).ToDictionary(
        pair => BigInteger.Parse((string)pair[0]!, CultureInfo.InvariantCulture), pair => (string)pair[1]!);
    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
        => throw new InvalidOperationException("Hungarian uses recursive group composition instead of card merging.");

    private string Cardinal(BigInteger value, bool omitZero)
    {
        if (value.Sign < 0) return "mínusz " + Cardinal(-value, false);
        if (value.IsZero) return omitZero ? "" : "nulla";
        if (omitZero && value == 2) return "két";
        if (value < 30) return _cards[value];
        if (value < 100) return _cards.TryGetValue(value, out string? exact) ? exact : _cards[value / 10 * 10] + Cardinal(value % 10, false);
        if (value < 1000) return (value / 100 == 1 ? "" : Cardinal(value / 100, true)) + "száz" + Cardinal(value % 100, true);
        if (value < 1000000)
        {
            string suffix = Cardinal(value % 1000, true);
            return (value / 1000 == 1 ? "" : Cardinal(value / 1000, true)) + "ezer"
                + (value <= 2000 || suffix.Length == 0 ? "" : "-") + suffix;
        }
        int digits = value.ToString(CultureInfo.InvariantCulture).Length;
        if (digits % 3 == 0) digits -= 2;
        BigInteger scale = BigInteger.Pow(10, digits / 3 * 3);
        string remainder = Cardinal(value % scale, true);
        return Cardinal(value / scale, true) + _cards[scale] + (remainder.Length == 0 ? "" : "-" + remainder);
    }

    public override string ToCardinal(BigInteger value) => Cardinal(value, false);
    public override string ToCardinal(double value)
    {
        if (value < 0) return "mínusz " + ToCardinal(-value);
        if (value == Math.Truncate(value)) return Cardinal(new BigInteger(value), false);
        string number = value.ToString("R", CultureInfo.InvariantCulture);
        int point = number.IndexOf('.');
        if (point < 0 || number.IndexOf('E') >= 0) throw new FormatException("Hungarian fractional conversion requires a decimal representation.");
        string right = number.Substring(point + 1);
        string unit = _cards[BigInteger.Pow(10, right.Length)];
        return Cardinal(new BigInteger(Math.Truncate(value)), false) + " egész " + Cardinal(BigInteger.Parse(right, CultureInfo.InvariantCulture), false)
            + " " + (string)LanguageData["partial_ordinals"]![unit]!;
    }

    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) return "mínusz " + ToOrdinal(-value);
        if (value.IsOne) return "első";
        if (value == 2) return "második";
        string text = Cardinal(value, false);
        foreach (var pair in (JObject)LanguageData["partial_ordinals"]!)
            if (text.EndsWith(pair.Key, StringComparison.Ordinal))
            { text = text.Substring(0, text.Length - pair.Key.Length) + (string)pair.Value!; break; }
        return text + "ik";
    }
}
