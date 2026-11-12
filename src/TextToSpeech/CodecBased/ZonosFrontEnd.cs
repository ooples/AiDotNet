using System.Globalization;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Zonos's text front end (<c>zonos/conditioning.py</c>'s <c>phonemize</c>): Japanese text normalized to its morphemes'
/// readings (<c>normalize_jp_text</c>), any other text's numbers spelled out (<c>normalize_numbers</c>, from VITS2), then
/// eSpeak's phonemes through phonemizer's backend with punctuation kept and stress marks kept.
/// </summary>
internal static class ZonosFrontEnd
{
    /// <summary>The punctuation phonemizer keeps (<c>_punctuation</c>).</summary>
    internal const string Punctuation = ";:,.!?¡¿—…\"«»“”() *~-/\\&";

    private static readonly Regex CommaNumber = new("([0-9][0-9,]+[0-9])", RegexOptions.CultureInvariant);
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Pounds = new("£([0-9,]*[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Dollars = new(@"\$([0-9.,]*[0-9]+)", RegexOptions.CultureInvariant);
    private static readonly Regex Ordinal = new("[0-9]+(st|nd|rd|th)", RegexOptions.CultureInvariant);
    private static readonly Regex Number = new("[0-9]+", RegexOptions.CultureInvariant);

    private static string Words(System.Numerics.BigInteger n) => InflectNumberWords.NumberToWords(n.ToString(CultureInfo.InvariantCulture));

    private static string ExpandDollars(Match m)
    {
        string match = m.Groups[1].Value;
        var parts = match.Split('.');
        if (parts.Length > 2) return match + " dollars";
        var dollars = parts[0].Length > 0 ? System.Numerics.BigInteger.Parse(parts[0], CultureInfo.InvariantCulture) : 0;
        var cents = parts.Length > 1 && parts[1].Length > 0 ? System.Numerics.BigInteger.Parse(parts[1], CultureInfo.InvariantCulture) : 0;
        string Unit(System.Numerics.BigInteger n, string one, string many) => n == 1 ? one : many;
        if (!dollars.IsZero && !cents.IsZero) return $"{dollars} {Unit(dollars, "dollar", "dollars")}, {cents} {Unit(cents, "cent", "cents")}";
        if (!dollars.IsZero) return $"{dollars} {Unit(dollars, "dollar", "dollars")}";
        if (!cents.IsZero) return $"{cents} {Unit(cents, "cent", "cents")}";
        return "zero dollars";
    }

    private static string ExpandNumber(Match m)
    {
        var num = System.Numerics.BigInteger.Parse(m.Value, CultureInfo.InvariantCulture);
        if (num > 1000 && num < 3000)
        {
            if (num == 2000) return "two thousand";
            if (num > 2000 && num < 2010) return "two thousand " + Words(num % 100);
            if (num % 100 == 0) return Words(num / 100) + " hundred";
            return InflectNumberWords.NumberToWords(num.ToString(CultureInfo.InvariantCulture), group: 2, andword: "", zero: "oh").Replace(", ", " ");
        }
        return InflectNumberWords.NumberToWords(num.ToString(CultureInfo.InvariantCulture), andword: "");
    }

    /// <summary><c>normalize_numbers</c>: commas dropped from numbers, pounds and dollars spelled out, decimals read with
    /// "point", ordinals and numbers spelled out (years 1001–2999 read in pairs).</summary>
    internal static string NormalizeNumbers(string text)
    {
        text = CommaNumber.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = Pounds.Replace(text, "$1 pounds");
        text = Dollars.Replace(text, ExpandDollars);
        text = DecimalNumber.Replace(text, m => m.Groups[1].Value.Replace(".", " point "));
        text = Ordinal.Replace(text, m => InflectNumberWords.NumberToWords(m.Value));
        text = Number.Replace(text, ExpandNumber);
        return text;
    }

    /// <summary>
    /// The phonemes of <paramref name="text"/> in <paramref name="language"/>: normalized (Japanese by
    /// <paramref name="japanese"/>, others by <see cref="NormalizeNumbers"/>), then phonemized by eSpeak through
    /// <paramref name="espeak"/> (language, text → eSpeak's phonemes; null for the installed library).
    /// </summary>
    internal static string Phonemize(string text, string language, Func<string, string> japanese, Func<string, string, string>? espeak)
    {
        string clean = language.Contains("ja") ? japanese(text) : NormalizeNumbers(text);
        return new EspeakPhonemizer(language, Punctuation, espeak, withStress: true).Phonemize(clean, "", " ");
    }
}
