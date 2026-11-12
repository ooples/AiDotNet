using System.Collections.Generic;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Reads numbers as English words the way espeak-ng's American English voice does.
/// </summary>
/// <remarks>
/// <para>Words joined by '+' are written by espeak as one word ("one+hundred"), so they share no word separator. Tens
/// and units are separate words ("twenty four"); there is no "and". Numbers from 1900 to 1999 are read as years
/// ("nineteen+hundred eight"); other four-digit numbers as thousands ("one thousand eight+hundred").</para>
/// <para>espeak names groups up to billions; numbers of a trillion or more are read digit by digit.</para>
/// </remarks>
internal static class EnglishNumberReader
{
    private static readonly string[] Ones =
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    };

    private static readonly string[] Tens =
        { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

    private static readonly Dictionary<string, string> IrregularOrdinals = new(System.StringComparer.Ordinal)
    {
        ["one"] = "first", ["two"] = "second", ["three"] = "third", ["five"] = "fifth", ["eight"] = "eighth",
        ["nine"] = "ninth", ["twelve"] = "twelfth",
    };

    /// <summary>The words of a cardinal number written in digits.</summary>
    public static List<string> Cardinal(string digits)
    {
        // Any Unicode decimal digit reads as its value, as Python's int() reads it.
        var ascii = new char[digits.Length];
        for (int i = 0; i < digits.Length; i++) ascii[i] = (char)('0' + (int)char.GetNumericValue(digits[i]));
        digits = new string(ascii);
        if (digits.Length > 12 || !long.TryParse(digits, System.Globalization.NumberStyles.None,
                System.Globalization.CultureInfo.InvariantCulture, out long n) || n >= 1_000_000_000_000L)
        {
            var spelled = new List<string>(digits.Length);
            foreach (char c in digits) spelled.Add(Ones[c - '0']);
            return spelled;
        }
        return Cardinal(n);
    }

    /// <summary>The words of an ordinal number written in digits ("29" gives "twenty ninth").</summary>
    public static List<string> Ordinal(string digits)
    {
        var words = Cardinal(digits);
        string lastWord = words[words.Count - 1];
        int join = lastWord.LastIndexOf('+');
        string last = join < 0 ? lastWord : lastWord.Substring(join + 1);
        string ordinal = IrregularOrdinals.TryGetValue(last, out var irregular) ? irregular
            : last.EndsWith("y", System.StringComparison.Ordinal) ? last.Substring(0, last.Length - 1) + "ieth"
            : last + "th";
        words[words.Count - 1] = join < 0 ? ordinal : lastWord.Substring(0, join + 1) + ordinal;
        return words;
    }

    private static List<string> Cardinal(long n)
    {
        if (n == 0) return new List<string> { "zero" };
        if (n >= 1900 && n <= 1999)
        {
            var year = new List<string> { "nineteen+hundred" };
            if (n % 100 != 0) year.AddRange(Below100((int)(n % 100)));
            return year;
        }
        var words = new List<string>();
        foreach (var (value, name) in new[] { (1_000_000_000L, "billion"), (1_000_000L, "million"), (1000L, "thousand") })
        {
            if (n >= value)
            {
                words.AddRange(Below1000((int)(n / value)));
                words.Add(name);
                n %= value;
            }
        }
        if (n > 0) words.AddRange(Below1000((int)n));
        return words;
    }

    private static List<string> Below100(int n)
    {
        if (n < 20) return new List<string> { Ones[n] };
        var words = new List<string> { Tens[n / 10] };
        if (n % 10 != 0) words.Add(Ones[n % 10]);
        return words;
    }

    private static List<string> Below1000(int n)
    {
        var words = new List<string>();
        if (n >= 100)
        {
            words.Add(Ones[n / 100] + "+hundred");
            n %= 100;
        }
        if (n > 0) words.AddRange(Below100(n));
        return words;
    }
}
