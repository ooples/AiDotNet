using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// cn2an 0.5.22's <c>transform(text, "an2cn")</c> (MIT): Arabic numerals in running text become Chinese — dates (a year
/// digit by digit, month and day as numbers), fractions (N/D as "D分之N"), percentages ("百分之…"), degrees Celsius
/// ("…摄氏度") and other numbers (signed, with decimals), each in cn2an's "low" style. A piece cn2an cannot convert is
/// left as written, as cn2an leaves it after its warning.
/// </summary>
internal static class Cn2An
{
    private static readonly string[] Digits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };

    private static readonly Regex Date = new(@"(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?", RegexOptions.CultureInvariant);
    private static readonly Regex Fraction = new(@"\d+/\d+", RegexOptions.CultureInvariant);
    private static readonly Regex Percent = new(@"-?(\d+\.)?\d+%", RegexOptions.CultureInvariant);
    private static readonly Regex Celsius = new(@"\d+℃", RegexOptions.CultureInvariant);
    private static readonly Regex Number = new(@"-?(\d+\.)?\d+", RegexOptions.CultureInvariant);
    private static readonly Regex Digit = new(@"\d+", RegexOptions.CultureInvariant);
    private static readonly Regex YearDigits = new(@"\d+(?=年)", RegexOptions.CultureInvariant);

    /// <summary><c>transform(text, "an2cn")</c>.</summary>
    public static string Transform(string text)
    {
        text = Date.Replace(text, m => Safely(m.Value, s => Digit.Replace(YearDigits.Replace(s, y => AnToCn(y.Value, direct: true)), d => AnToCn(d.Value))));
        text = Fraction.Replace(text, m => Safely(m.Value, s =>
        {
            var parts = Digit.Replace(s, d => AnToCn(d.Value)).Split('/');
            return parts[1] + "分之" + parts[0];
        }));
        text = Percent.Replace(text, m => Safely(m.Value, s => "百分之" + AnToCn(s.Substring(0, s.Length - 1))));
        text = Celsius.Replace(text, m => Safely(m.Value, s => AnToCn(s.Substring(0, s.Length - 1)) + "摄氏度"));
        return Number.Replace(text, m => Safely(m.Value, s => AnToCn(s)));
    }

    // __sub_util: an empty match becomes nothing; a failed conversion leaves the piece as it was.
    private static string Safely(string piece, Func<string, string> convert)
    {
        if (piece.Length == 0) return "";
        try
        {
            return convert(piece);
        }
        catch (ArgumentException)
        {
            return piece;
        }
    }

    /// <summary><c>An2Cn().an2cn(number, "low")</c>, or <c>"direct"</c> (each digit, '.' as 点): full-width characters made
    /// half-width first, anything but ASCII digits, '.' and '-' refused.</summary>
    public static string AnToCn(string number, bool direct = false)
    {
        var half = new StringBuilder(number.Length);
        foreach (char c in number) half.Append(c == '　' ? ' ' : c >= '！' && c <= '～' ? (char)(c - 0xfee0) : c);
        number = half.ToString();
        foreach (char c in number)
            if (!(c is >= '0' and <= '9' || c == '.' || c == '-'))
                throw new ArgumentException($"'{number}' is outside cn2an's range.", nameof(number));
        if (!direct) return MandarinG2P.AnToCn(number, joinThousands: false);
        string sign = number.StartsWith("-", StringComparison.Ordinal) ? "负" : "";
        var output = new StringBuilder(sign);
        foreach (char c in sign.Length > 0 ? number.Substring(1) : number)
        {
            if (c == '.') output.Append('点');
            else if (c == '-') throw new ArgumentException($"'{number}' is outside cn2an's range.", nameof(number));
            else output.Append(Digits[c - '0']);
        }
        return output.ToString();
    }
}
