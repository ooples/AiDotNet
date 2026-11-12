using System.Numerics;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

internal abstract class Num2WordsEuropean : Num2WordsBase
{
    private static readonly Lazy<JObject> Data = new(() => JObject.Parse(string.Join("\n", ChineseText.ReadLines("num2words_european.json.gz"))));
    private readonly string _language;
    protected Num2WordsEuropean(string language, bool currencyForms = true) : base(((JArray)Data.Value[language]!["cards"]!).Select(entry =>
        (BigInteger.Parse((string)entry[0]!, System.Globalization.CultureInfo.InvariantCulture), (string)entry[1]!)),
        (string)Data.Value[language]!["negative"]!, (string)Data.Value[language]!["point"]!, " ", currencyForms ? language : null)
        => _language = language;
    protected JToken LanguageData => Data.Value[_language]!;

    protected string? OrdinalWord(string word)
        => (string?)((JArray)Data.Value[_language]!["ordinals"]!).FirstOrDefault(pair => (string)pair[0]! == word)?[1];

    protected string OrdinalStem(string word)
    {
        foreach (var pair in (JArray)Data.Value[_language]!["ordinals"]!)
        {
            string ending = (string)pair[0]!;
            if (word.EndsWith(ending, StringComparison.Ordinal))
                return word.Substring(0, word.Length - ending.Length) + (string)pair[1]!;
        }
        return word;
    }
}

/// <summary>num2words 0.5.12's German merge, ordinal and currency rules.</summary>
internal sealed class Num2WordsGerman : Num2WordsEuropean
{
    private static readonly Lazy<Num2WordsGerman> Shared = new(() => new Num2WordsGerman());
    internal static Num2WordsGerman Default => Shared.Value;
    private Num2WordsGerman() : base("de") { }

    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
    {
        if (l.IsOne)
        {
            if (r == 100 || r == 1000) return ("ein" + right, r);
            if (r < 1000000) return (right, r);
            left = "eine";
        }
        BigInteger value;
        if (r > l)
        {
            if (r >= 1000000)
            {
                if (l > 1) right += right.EndsWith("e", StringComparison.Ordinal) ? "n" : "en";
                left += " ";
            }
            value = l * r;
        }
        else
        {
            if (r < 10 && 10 < l && l < 100)
            {
                if (r.IsOne) right = "ein";
                (right, left) = (left, right + "und");
            }
            else if (l >= 1000000) left += " ";
            value = l + r;
        }
        return (left + right, value);
    }

    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        string result = OrdinalStem(ToCardinal(value).ToLowerInvariant()) + "te";
        if (result is "eintausendste" or "einhundertste") result = result.Substring(3);
        result = Regex.Replace(result, @"eine ([a-z]+(illion|illiard)ste)$", match => match.Groups[1].Value);
        return Regex.Replace(result, @" ([a-z]+(illion|illiard)ste)$", match => match.Groups[1].Value);
    }

    protected override string FinishCurrency(string result, string currency, string separator) => result.Replace("eins ", "ein ");
}

/// <summary>num2words 0.5.12's Dutch merge, ordinal and fixed-singular currency rules.</summary>
internal sealed class Num2WordsDutch : Num2WordsEuropean
{
    private static readonly Lazy<Num2WordsDutch> Shared = new(() => new Num2WordsDutch());
    internal static Num2WordsDutch Default => Shared.Value;
    private Num2WordsDutch() : base("nl") { }

    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
    {
        if (l.IsOne)
        {
            if (r < 1000000) return (right, r);
            left = "een";
        }
        BigInteger value;
        if (r > l)
        {
            if (r >= 1000000) left += " ";
            value = l * r;
        }
        else
        {
            if (r < 10 && 10 < l && l < 100)
            {
                if (r.IsOne) right = "een";
                right += right.EndsWith("e", StringComparison.Ordinal) ? "ën" : "en";
                (right, left) = (left, right);
            }
            else if (l >= 1000000) left += " ";
            value = l + r;
        }
        return (left + right, value);
    }

    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        return OrdinalStem(ToCardinal(value)) + "e";
    }

    protected override string Pluralize(BigInteger n, string[] forms) => forms[0];
}
