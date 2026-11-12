using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's recursive Italian cardinal/ordinal algorithm and vowel contractions.</summary>
internal sealed class Num2WordsItalian : Num2WordsBase
{
    private static readonly string[] Cardinals = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici", "diciassette", "diciotto", "diciannove"];
    private static readonly string[] Ordinals = ["zero", "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono", "decimo", "undicesimo", "dodicesimo", "tredicesimo", "quattordicesimo", "quindicesimo", "sedicesimo", "diciassettesimo", "diciottesimo", "diciannovesimo"];
    private static readonly string[] HighPrefixes = ["zero", "m", "b", "tr", "quadr", "quint", "sest", "sett", "ott", "nov", "dec"];
    private static readonly Lazy<Num2WordsItalian> Shared = new(() => new Num2WordsItalian());
    internal static Num2WordsItalian Default => Shared.Value;
    private Num2WordsItalian() : base(Enumerable.Range(0, 20).Reverse().Select(i => (new BigInteger(i), Cardinals[i])), "meno ", "virgola", " ", "it") { }
    // The original Italian converter bypasses the shared card-merging algorithm, but reuses its currency machinery.
    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
        => throw new InvalidOperationException("Italian uses recursive cardinal conversion instead of card merging.");

    private static string Contract(string text) => text.Replace("oo", "o").Replace("ao", "o").Replace("io", "o").Replace("au", "u").Replace("iu", "u");
    private static string Accent(string text) => string.Join(" ", Regex.Split(text.Trim(), @"\s+").Select(word =>
        word.EndsWith("tre", StringComparison.Ordinal) && word.Length > 3
            ? word.Replace("tré", "tre").Substring(0, word.Length - 3) + "tré" : word.Replace("tré", "tre")));
    private string OmitZero(BigInteger value) => value.IsZero ? "" : ToCardinal(value);

    public override string ToCardinal(BigInteger value)
    {
        string text;
        if (value.Sign < 0) text = "meno " + ToCardinal(-value);
        else if (value < 20) text = Cardinals[(int)value];
        else if (value < 100)
        {
            int tens = (int)(value / 10);
            string prefix = tens == 2 ? "venti" : tens == 3 ? "trenta" : tens == 4 ? "quaranta" : tens == 6 ? "sessanta"
                : Cardinals[tens].Substring(0, Cardinals[tens].Length - 1) + "anta";
            text = Contract(prefix + OmitZero(value % 10));
        }
        else if (value < 1000)
            text = Contract((value / 100 == 1 ? "" : Cardinals[(int)(value / 100)]) + "cento" + OmitZero(value % 100));
        else if (value < 1000000)
            text = (value / 1000 == 1 ? "mille" : ToCardinal(value / 1000) + "mila") + OmitZero(value % 1000);
        else
        {
            int length = value.ToString(CultureInfo.InvariantCulture).Length;
            if (length >= 66) throw new NotSupportedException("The Italian converter accepts at most 65 digits.");
            int preceding = length % 3; if (preceding == 0) preceding = 3;
            int exponent = length - preceding;
            BigInteger scale = BigInteger.Pow(10, exponent), multiplier = value / scale, remainder = value % scale;
            string infix = HighPrefixes[exponent / 6] + (exponent % 6 == 0 ? "ilione" : "iliardo");
            string prefix = multiplier.IsOne ? "un " : ToCardinal(multiplier);
            if (!multiplier.IsOne) infix = " " + infix.Substring(0, infix.Length - 1) + "i";
            string suffix = OmitZero(remainder);
            if (suffix.Length > 0) infix += suffix.Contains(" e ") ? ", " : " e ";
            text = prefix + infix + suffix;
        }
        return Accent(text);
    }

    public override string ToCardinal(double value)
    {
        if (value < 0) return Accent("meno " + ToCardinal(-value));
        if (double.IsNaN(value) || double.IsInfinity(value) || (value != 0 && (value < .0001 || value >= 1e16)))
            throw new FormatException("The original Italian float reader requires a non-scientific decimal representation.");
        string digits = value.ToString("R", CultureInfo.InvariantCulture);
        string fraction = digits.IndexOf('.') < 0 ? "0" : digits.Substring(digits.IndexOf('.') + 1);
        return Accent(ToCardinal(new BigInteger(Math.Truncate(value))) + " virgola "
            + string.Join(" ", fraction.Select(c => ToCardinal(new BigInteger(c - '0')))));
    }

    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) return "meno " + ToOrdinal(-value);
        if (value < 20) return Ordinals[(int)value];
        bool outsideTeens = !(10 < value % 100 && value % 100 < 20);
        string cardinal = ToCardinal(value);
        if (outsideTeens && value % 10 == 3) return cardinal.Substring(0, cardinal.Length - 1) + "eesimo";
        if (outsideTeens && value % 10 == 6) return cardinal + "esimo";
        string prefix = cardinal.Substring(0, cardinal.Length - 1);
        if (prefix.EndsWith("mil", StringComparison.Ordinal)) prefix += "l";
        return prefix + "esimo";
    }

    protected override string FinishCurrency(string result, string currency, string separator)
    {
        if (currency == "GBP" && result.StartsWith("uno ", StringComparison.Ordinal)) result = "una " + result.Substring(4);
        return result.Replace("uno", "un");
    }
}
