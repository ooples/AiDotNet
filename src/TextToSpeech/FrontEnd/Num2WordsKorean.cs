using System.Numerics;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's Korean cardinal, ordinal and currency forms.</summary>
internal sealed class Num2WordsKorean : Num2WordsEuropean
{
    private static readonly Lazy<Num2WordsKorean> Shared = new(() => new Num2WordsKorean());
    internal static Num2WordsKorean Default => Shared.Value;
    private Num2WordsKorean() : base("ko", currencyForms: false) { }

    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
    {
        if (l.IsOne && r <= 10000) return (right, r);
        if (10000 > l && l > r) return (left + right, l + r);
        if (l >= 10000 && l > r) return (left + " " + right, l + r);
        return (left + right, l * r);
    }

    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        if (value.IsOne) return "첫 번째";
        var words = ToCardinal(value).Split(' ');
        var hundreds = words[words.Length - 1].Split('백');
        int last = hundreds.Length - 1;
        if (hundreds[last].Contains("십"))
        {
            var pieces = hundreds[last].Split('십');
            pieces[0] = OrdinalWord(pieces[0] + "십") ?? throw new KeyNotFoundException("Unsupported Korean ordinal stem.");
            string? unit = OrdinalWord(pieces[1]);
            if (unit is not null) { pieces[1] = unit; pieces[0] = pieces[0].Replace("스무", "스물"); }
            hundreds[last] = string.Concat(pieces);
        }
        else hundreds[last] = OrdinalWord(hundreds[last]) ?? throw new KeyNotFoundException("Unsupported Korean ordinal stem.");
        words[words.Length - 1] = string.Join("백 ", hundreds);
        return string.Join(" ", words) + " 번째";
    }

    public override string ToCurrency(double value, string currency, string separator)
    {
        var (units, cents, negative) = ParseCurrencyParts(value);
        string word = currency == "USD" ? "달러" : currency == "KRW" ? "원" : currency == "JPY" ? "엔"
            : throw new NotSupportedException($"Korean currency '{currency}' is unsupported by num2words.");
        if (currency != "USD" && !cents.IsZero) throw new NotSupportedException("This Korean currency has no fractional unit.");
        return (negative ? "마이너스 " : "") + ToCardinal(units).Replace(" ", "") + word
            + (currency == "USD" ? " " + ToCardinal(cents) + "센트" : "");
    }
}
