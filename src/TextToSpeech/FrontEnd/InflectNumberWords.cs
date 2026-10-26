using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The Python <c>inflect</c> package's <c>number_to_words</c> (version 7.0.0, MIT), which the MeloTTS and g2p_en text
/// normalizers use: "1234" gives "one thousand, two hundred and thirty-four", with the "and" word, the zero word and the
/// grouping configurable, and an ordinal suffix ("21st") giving the ordinal ("twenty-first").
/// </summary>
internal static class InflectNumberWords
{
    private static readonly string[] Unit = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
    private static readonly string[] Teen = { "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
    private static readonly string[] Ten = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
    private static readonly string[] Mill =
    {
        " ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion", " septillion",
        " octillion", " nonillion", " decillion",
    };
    private static readonly HashSet<string> NthSuffixes = new(StringComparer.Ordinal) { "th", "st", "nd", "rd" };
    private static readonly (string Ending, string Ordinal)[] Ordinals =
    {
        ("ty", "tieth"), ("one", "first"), ("two", "second"), ("three", "third"), ("five", "fifth"), ("eight", "eighth"),
        ("nine", "ninth"), ("twelve", "twelfth"),
    };

    private static readonly Regex DigitGroup = new(@"(\d)", RegexOptions.CultureInvariant);
    private static readonly Regex TwoDigits = new(@"(\d)(\d)", RegexOptions.CultureInvariant);
    private static readonly Regex ThreeDigits = new(@"(\d)(\d)(\d)", RegexOptions.CultureInvariant);
    private static readonly Regex ThreeDigitsWord = new(@"(\d)(\d)(\d)(?=\D*\z)", RegexOptions.CultureInvariant);
    private static readonly Regex TwoDigitsWord = new(@"(\d)(\d)(?=\D*\z)", RegexOptions.CultureInvariant);
    private static readonly Regex OneDigitWord = new(@"(\d)(?=\D*\z)", RegexOptions.CultureInvariant);
    private static readonly Regex NonDigit = new(@"\D", RegexOptions.CultureInvariant);
    private static readonly Regex WhitespacesComma = new(@"\s+,", RegexOptions.CultureInvariant);
    private static readonly Regex CommaWord = new(@", (\S+)\s+\z", RegexOptions.CultureInvariant);
    private static readonly Regex Whitespaces = new(@"\s+", RegexOptions.CultureInvariant);

    private sealed class Arguments
    {
        public string And = "and", Zero = "zero", One = "one";
        public int MillCount;
    }

    private static string MillFn(int index)
    {
        if (index > Mill.Length - 1) throw new ArgumentOutOfRangeException(nameof(index), "The number is out of inflect's range.");
        return Mill[index];
    }

    private static string UnitFn(int units, int mindex) => Unit[units] + MillFn(mindex);

    private static string TenFn(int tens, int units, int mindex = 0)
    {
        if (tens != 1)
            return Ten[tens] + (tens != 0 && units != 0 ? "-" : "") + Unit[units] + MillFn(mindex);
        return Teen[units] + Mill[mindex];
    }

    private static string HundFn(Arguments a, int hundreds, int tens, int units, int mindex)
    {
        if (hundreds != 0)
        {
            string andword = tens != 0 || units != 0 ? $" {a.And} " : "";
            return $"{Unit[hundreds]} hundred{andword}{TenFn(tens, units)}{MillFn(mindex)}, ";
        }
        if (tens != 0 || units != 0) return $"{TenFn(tens, units)}{MillFn(mindex)}, ";
        return "";
    }

    private static int D(Match m, int group) => m.Groups[group].Value[0] - '0';

    private static string Enword(Arguments a, string num, int group)
    {
        if (group == 1)
            return DigitGroup.Replace(num, m => D(m, 1) == 1 ? $" {a.One}, " : D(m, 1) != 0 ? $"{Unit[D(m, 1)]}, " : $" {a.Zero}, ");
        if (group == 2)
        {
            num = TwoDigits.Replace(num, m => Group2(a, m));
            return DigitGroup.Replace(num, m => D(m, 1) != 0 ? $"{Unit[D(m, 1)]}, " : $" {a.Zero}, ", 1);
        }
        if (group == 3)
        {
            num = ThreeDigits.Replace(num, m =>
            {
                int hundreds = D(m, 1), tens = D(m, 2), units = D(m, 3);
                string hunword = hundreds == 1 ? $" {a.One}" : hundreds != 0 ? Unit[hundreds] : $" {a.Zero}";
                string tenword = tens != 0 ? TenFn(tens, units) : units != 0 ? $" {a.Zero} {Unit[units]}" : $" {a.Zero} {a.Zero}";
                return $"{hunword} {tenword}, ";
            });
            num = TwoDigits.Replace(num, m => Group2(a, m), 1);
            return DigitGroup.Replace(num, m => D(m, 1) == 1 ? $" {a.One}, " : D(m, 1) != 0 ? $"{Unit[D(m, 1)]}, " : $" {a.Zero}, ", 1);
        }
        var value = System.Numerics.BigInteger.Parse(num.Trim());
        if (value.IsZero) return a.Zero;
        if (value.IsOne) return a.One;
        num = num.TrimStart().TrimStart('0');
        a.MillCount = 0;
        while (ThreeDigitsWord.IsMatch(num))
            num = ThreeDigitsWord.Replace(num, m =>
            {
                string r = HundFn(a, D(m, 1), D(m, 2), D(m, 3), a.MillCount);
                a.MillCount++;
                return r;
            }, 1);
        num = TwoDigitsWord.Replace(num, m => $"{TenFn(D(m, 1), D(m, 2), a.MillCount)}, ", 1);
        return OneDigitWord.Replace(num, m => $"{UnitFn(D(m, 1), a.MillCount)}, ", 1);
    }

    private static string Group2(Arguments a, Match m)
    {
        int tens = D(m, 1), units = D(m, 2);
        if (tens != 0) return $"{TenFn(tens, units)}, ";
        if (units != 0) return $" {a.Zero} {Unit[units]}, ";
        return $" {a.Zero} {a.Zero}, ";
    }

    /// <summary><c>inflect.engine().ordinal(words)</c> for a number in words (<c>_sub_ord</c>): the last word's ending
    /// made ordinal ("three" → "third", "twenty" → "twentieth"), or "th" appended.</summary>
    public static string Ordinal(string words)
    {
        var match = Ordinals.FirstOrDefault(o => words.EndsWith(o.Ending, StringComparison.Ordinal));
        return match.Ending is not null ? words.Substring(0, words.Length - match.Ending.Length) + match.Ordinal : words + "th";
    }

    /// <summary><c>inflect.engine().number_to_words(num, group=…, comma=…, andword=…, zero=…, one=…, decimal=…)</c>.</summary>
    public static string NumberToWords(string num, int group = 0, string comma = ",", string andword = "and", string zero = "zero",
        string one = "one", string? @decimal = "point")
    {
        if (group < 0 || group > 3) throw new ArgumentOutOfRangeException(nameof(group));
        var a = new Arguments { And = andword, Zero = zero, One = one };
        string nowhite = num.TrimStart();
        string sign = nowhite.Length > 0 && nowhite[0] == '+' ? "plus" : nowhite.Length > 0 && nowhite[0] == '-' ? "minus" : "";
        if (NthSuffixes.Contains(num)) num = zero;
        bool ordinal = num.Length >= 2 && NthSuffixes.Contains(num.Substring(num.Length - 2));
        if (ordinal) num = num.Substring(0, num.Length - 2);
        bool finalPoint = false;
        List<string> chunks;
        if (!string.IsNullOrEmpty(@decimal))
        {
            chunks = group != 0 ? num.Split('.').ToList() : num.Split(new[] { '.' }, 2).ToList();
            if (chunks[chunks.Count - 1] == "")
            {
                chunks.RemoveAt(chunks.Count - 1);
                finalPoint = true;
            }
        }
        else
        {
            chunks = new List<string> { num };
        }

        // first: 1 initially, "" after the first chunk, 0 when there is no integer part.
        int firstState = 1;
        int loopStart = 0;
        if (chunks.Count > 0 && chunks[0] == "")
        {
            firstState = 0;
            if (chunks.Count > 1) loopStart = 1;
        }
        for (int i = loopStart; i < chunks.Count; i++)
        {
            string chunk = NonDigit.Replace(chunks[i], "");
            if (chunk == "") chunk = "0";
            chunk = group == 0 && firstState != 1 ? Enword(a, chunk, 1) : Enword(a, chunk, group);
            if (chunk.EndsWith(", ", StringComparison.Ordinal)) chunk = chunk.Substring(0, chunk.Length - 2);
            chunk = WhitespacesComma.Replace(chunk, ",");
            if (group == 0 && firstState == 1) chunk = CommaWord.Replace(chunk, m => $" {andword} {m.Groups[1].Value}");
            chunk = Whitespaces.Replace(chunk, " ").Trim();
            if (firstState == 1) firstState = 2;                      // Python's first = ""
            chunks[i] = chunk;
        }

        var numchunks = new List<string>();
        if (firstState != 0) numchunks = chunks[0].Split(new[] { comma + " " }, StringSplitOptions.None).ToList();
        if (ordinal && numchunks.Count > 0)
        {
            string last = numchunks[numchunks.Count - 1];
            var match = Ordinals.FirstOrDefault(o => last.EndsWith(o.Ending, StringComparison.Ordinal));
            numchunks[numchunks.Count - 1] = match.Ending is not null
                ? last.Substring(0, last.Length - match.Ending.Length) + match.Ordinal
                : last + "th";
        }
        foreach (string chunk in chunks.Skip(1))
        {
            numchunks.Add(@decimal!);
            numchunks.AddRange(chunk.Split(new[] { comma + " " }, StringSplitOptions.None));
        }
        if (finalPoint) numchunks.Add(@decimal!);

        string signout = sign.Length > 0 ? sign + " " : "";
        if (group != 0) return signout + string.Join(", ", numchunks);
        string result = signout + numchunks[0];
        bool first = @decimal is null || !result.EndsWith(@decimal, StringComparison.Ordinal);
        foreach (string nc in numchunks.Skip(1))
        {
            if (nc == @decimal)
            {
                result += " " + nc;
                first = false;
            }
            else if (first)
            {
                result += comma + " " + nc;
            }
            else
            {
                result += " " + nc;
            }
        }
        return result;
    }
}
