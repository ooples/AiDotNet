using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Coqui XTTS's ordered English number, abbreviation and symbol expansions.</summary>
internal static class XttsLatinCleaner
{
    private static readonly Lazy<JObject> Rules = new(() => JObject.Parse(string.Join("\n", ChineseText.ReadLines("xtts_cleaner_rules.json.gz"))));

    internal static string English(string text)
        => Numbers(text, "en", Num2WordsEnglish.Default, Num2WordsEnglish.Default.ToOrdinal);

    internal static string French(string text)
        => Numbers(text, "fr", Num2WordsFrench.Default, Num2WordsFrench.Default.ToOrdinal);

    internal static string Spanish(string text)
        => Numbers(text, "es", Num2WordsSpanish.Default, Num2WordsSpanish.Default.ToOrdinal);

    internal static string German(string text)
        => Numbers(text, "de", Num2WordsGerman.Default, Num2WordsGerman.Default.ToOrdinal);

    internal static string Dutch(string text)
        => Numbers(text, "nl", Num2WordsDutch.Default, Num2WordsDutch.Default.ToOrdinal);

    internal static string Korean(string text)
        => Numbers(text, "ko", Num2WordsKorean.Default, Num2WordsKorean.Default.ToOrdinal);

    internal static string Portuguese(string text)
        => Numbers(text, "pt", Num2WordsPortuguese.Default, Num2WordsPortuguese.Default.ToOrdinal);

    internal static string Italian(string text)
        => Numbers(text, "it", Num2WordsItalian.Default, Num2WordsItalian.Default.ToOrdinal);

    internal static string Russian(string text)
        => Numbers(text, "ru", Num2WordsSlavic.Russian, Num2WordsSlavic.Russian.ToOrdinal);
    internal static string Polish(string text)
        => Numbers(text, "pl", Num2WordsSlavic.Polish, Num2WordsSlavic.Polish.ToOrdinal);
    internal static string Czech(string text)
        => Numbers(text, "cs", Num2WordsSlavic.Czech, Num2WordsSlavic.Czech.ToOrdinal);
    internal static string Hungarian(string text)
        => Numbers(text, "hu", Num2WordsHungarian.Default, Num2WordsHungarian.Default.ToOrdinal);
    internal static string Turkish(string text)
        => Numbers(text, "tr", Num2WordsTurkish.Default, Num2WordsTurkish.Default.ToOrdinal);
    internal static string Arabic(string text)
    {
        // Keep the source converter's intra-utterance gender transitions, while isolating unrelated utterances.
        var reader = new Num2WordsArabic();
        return Numbers(text, "ar", reader, reader.ToOrdinal);
    }

    private static string Numbers(string text, string language, Num2WordsBase numbers, Func<BigInteger, string> ordinal)
    {
        text = language is "en" or "ru"
            ? Regex.Replace(text, @"\b\d{1,3}(,\d{3})*(\.\d+)?\b", match => match.Value.Replace(",", ""))
            : Regex.Replace(text, @"\b\d{1,3}(.\d{3})*(\,\d+)?\b", match => match.Value.Replace(".", ""));
        string Currency(Match match, string code)
        {
            double amount = double.Parse(Regex.Replace(match.Value.Replace(",", "."), @"[^\d.]", ""), CultureInfo.InvariantCulture);
            string expanded = numbers.ToCurrency(amount, code, language == "fr" ? " et" : language == "es" ? " con" : language == "de" ? " und" : language is "pt" or "it" ? " e" : ",");
            if (amount == Math.Truncate(amount))
            {
                string separator = language == "es" ? " con " : language == "fr" ? " et " : language == "de" ? " und " : language is "pt" or "it" ? " e " : ", ";
                int last = expanded.LastIndexOf(separator, StringComparison.Ordinal);
                if (last >= 0) expanded = expanded.Substring(0, last);
            }
            return expanded;
        }
        try
        {
            text = Regex.Replace(text, @"((£[0-9.,]*[0-9]+)|([0-9.,]*[0-9]+£))", match => Currency(match, "GBP"));
            text = Regex.Replace(text, @"((\$[0-9.,]*[0-9]+)|([0-9.,]*[0-9]+\$))", match => Currency(match, "USD"));
            text = Regex.Replace(text, @"(([0-9.,]*[0-9]+€)|(€[0-9.,]*[0-9]+))", match => Currency(match, "EUR"));
        }
        catch (Exception error) when (error is FormatException or OverflowException or NotSupportedException)
        { /* The released cleaner retains malformed/unsupported currency text and proceeds with ordinary numbers. */ }
        if (language != "tr") text = Regex.Replace(text, @"([0-9]+[.,][0-9]+)", match => numbers.ToCardinal(double.Parse(match.Value.Replace(",", "."), CultureInfo.InvariantCulture)));
        string ordinalPattern = language == "fr" ? @"([0-9]+)(º|ª|er|re|e|ème)"
            : language == "es" ? @"([0-9]+)(º|ª|er|o|a|os|as)"
            : language == "de" ? @"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"
            : language == "nl" ? @"([0-9]+)(de|ste|e)"
            : language == "pt" ? @"([0-9]+)(º|ª|o|a|os|as)"
            : language == "it" ? @"([0-9]+)(º|°|ª|o|a|i|e)"
            : language == "pl" ? @"([0-9]+)(º|ª|st|nd|rd|th)"
            : language == "ru" ? @"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"
            : language == "cs" ? @"([0-9]+)\.(?=\s|$)"
            : language == "hu" ? @"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"
            : language == "tr" ? @"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"
            : language == "ar" ? @"([0-9]+)(ون|ين|ث|ر|ى)"
            : language == "ko" ? @"([0-9]+)(번째|번|차|째)" : @"([0-9]+)(st|nd|rd|th)";
        text = Regex.Replace(text, ordinalPattern, match => ordinal(BigInteger.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture)));
        text = Regex.Replace(text, @"[0-9]+", match => numbers.ToCardinal(BigInteger.Parse(match.Value, CultureInfo.InvariantCulture)));
        return ApplyLexicalRules(text, language);
    }

    internal static string ApplyLexicalRules(string text, string language)
    {
        var rules = Rules.Value[language] ?? throw new NotSupportedException($"No XTTS cleaner rules for '{language}'.");
        foreach (string group in new[] { "abbreviations", "symbols" })
            foreach (var rule in (JArray)rules[group]!)
                text = Regex.Replace(text, (string)rule["pattern"]!, _ => (string)rule["replacement"]!,
                    (bool)rule["ignore_case"]! ? RegexOptions.IgnoreCase | RegexOptions.CultureInvariant : RegexOptions.CultureInvariant);
        return Regex.Replace(text, @"\s+", " ").Trim();
    }
}
