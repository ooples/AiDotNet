using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's Portuguese (Portugal) forms.</summary>
internal sealed class Num2WordsPortuguese : Num2WordsEuropean
{
    private static readonly Lazy<Num2WordsPortuguese> Shared = new(() => new Num2WordsPortuguese());
    internal static Num2WordsPortuguese Default => Shared.Value;
    private Num2WordsPortuguese() : base("pt") { }

    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
    {
        if (l.IsOne) { if (r < 1000000) return (right, r); left = "um"; }
        else if (l == 100 && r % 1000 != 0) left = "cento";
        if (r < l) return (left + " e " + right, l + r);
        if (r % 1000000000 == 0 && l > 1) right = right.Substring(0, right.Length - 4) + "liões";
        else if (r % 1000000 == 0 && l > 1) right = right.Substring(0, right.Length - 4) + "lhões";
        if (right == "milião") right = "milhão";
        if (r == 100) { left = (string)LanguageData["hundreds"]![l.ToString(CultureInfo.InvariantCulture)]!; right = ""; }
        else right = " " + right;
        return (left + right, l * r);
    }

    private static string FinishCardinal(string text)
    {
        foreach (string unit in new[] { "mil", "milhão", "milhões", "mil milhões", "bilião", "biliões", "mil biliões" })
            if (Regex.IsMatch(text, ".*" + unit + @" e \w*entos? (?=.*e)")) text = text.Replace(unit + " e", unit);
        return text;
    }
    public override string ToCardinal(BigInteger value) => FinishCardinal(base.ToCardinal(value));
    public override string ToCardinal(double value) => FinishCardinal(base.ToCardinal(value));

    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        string digits = value.ToString(CultureInfo.InvariantCulture), thousands = "";
        var parts = new List<string>();
        for (int i = 0; i < digits.Length; i++)
        {
            if (i > 0 && i % 3 == 0) thousands = (string?)LanguageData["thousands"]![i.ToString(CultureInfo.InvariantCulture)]
                ?? throw new NotSupportedException("Portuguese ordinal exceeds the num2words table.");
            int digit = digits[digits.Length - i - 1] - '0';
            if (digit != 0 && thousands.Length > 0) { parts.Add(thousands); thousands = ""; }
            parts.Add((string)LanguageData["ordinals"]![i % 3]![digit.ToString(CultureInfo.InvariantCulture)]!);
        }
        parts.Reverse(); string result = Regex.Replace(string.Join(" ", parts).Trim(), @"\s+", " ");
        return result.StartsWith("primeiro", StringComparison.Ordinal) && !value.IsOne ? result.Substring(9) : result;
    }

    protected override string FinishCurrency(string result, string currency, string separator)
    {
        if (result.StartsWith("menos  ", StringComparison.Ordinal)) result = "menos " + result.Substring(7);
        string plural = (string)CurrencyData.Value["pt"]![currency]![0]![1]!;
        foreach (string unit in new[] { "milhão", "milhões", "bilião", "biliões", "trilião", "triliões" })
            if (Regex.IsMatch(result, ".*" + unit + " (?=" + Regex.Escape(plural) + ")"))
            {
                int first = result.IndexOf(unit, StringComparison.Ordinal);
                result = result.Substring(0, first) + unit + " de" + result.Substring(first + unit.Length);
            }
        return result.Replace(" e zero cêntimos", "");
    }
}
