using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Python's string semantics where ported front ends depend on them: <c>str.isspace</c> (and so <c>\s</c> in a
/// <c>str</c> pattern), <c>str.strip</c>, <c>str.lower</c>, <c>str.isdigit</c> and <c>\w</c>.
/// </summary>
internal static class PythonText
{
    /// <summary>A regex character-class body matching what Python's <c>\s</c> does: .NET's plus U+001C to U+001F.</summary>
    public const string WhitespaceClass = @"\s\x1c-\x1f";

    /// <summary>A regex character-class body matching Python's <c>\w</c> (alphanumerics and underscore).</summary>
    public const string WordClass = @"\p{L}\p{N}_";

    public static bool IsSpace(char c) => char.IsWhiteSpace(c) || (c >= '\x1c' && c <= '\x1f');

    public static string Strip(string s)
    {
        int start = 0, end = s.Length;
        while (start < end && IsSpace(s[start])) start++;
        while (end > start && IsSpace(s[end - 1])) end--;
        return s.Substring(start, end - start);
    }

    /// <summary><c>str.isdigit</c>: non-empty, and every character a digit (decimal digits and the likes of ²).</summary>
    public static bool IsDigit(string s)
    {
        if (s.Length == 0) return false;
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length)
            {
                if (CharUnicodeInfo.GetDigitValue(s, i) < 0) return false;
                i++;
            }
            else if (CharUnicodeInfo.GetDigitValue(s[i]) < 0) return false;
        }
        return true;
    }

    /// <summary>
    /// <c>str.lower</c>: invariant lowercasing plus Unicode's special casing Python applies (İ becomes i + combining dot,
    /// and a capital sigma ending a word becomes final sigma).
    /// </summary>
    public static string Lower(string s)
    {
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c == 'İ') sb.Append("i̇");
            else if (c == 'Σ')
            {
                bool afterLetter = i > 0 && char.IsLetter(s[i - 1]);
                bool beforeLetter = i + 1 < s.Length && char.IsLetter(s[i + 1]);
                sb.Append(afterLetter && !beforeLetter ? 'ς' : 'σ');
            }
            else sb.Append(char.ToLowerInvariant(c));
        }
        return sb.ToString();
    }
}

/// <summary>
/// A value parsed by Python's <c>decimal.Decimal(str)</c>: a sign, a coefficient and a power-of-ten exponent, or an
/// infinity or (signalling) NaN. The grammar is the decimal module's: optional sign, digits with an optional point,
/// an optional exponent, underscores between digits, any Unicode decimal digits, surrounding whitespace ignored.
/// </summary>
internal readonly struct PythonDecimal
{
    public enum DecimalKind { Finite, Infinity, QuietNaN, SignalingNaN }

    private static readonly Regex FiniteForm = new(
        @"^(?<sign>[+-])?(?:(?<int>\d+(?:_\d+)*)(?:\.(?<frac>\d+(?:_\d+)*)?)?|\.(?<frac>\d+(?:_\d+)*))(?:[eE](?<exp>[+-]?\d+(?:_\d+)*))?$",
        RegexOptions.CultureInvariant);
    private static readonly Regex InfinityForm = new(@"^[+-]?(?:inf|infinity)$", RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
    private static readonly Regex NaNForm = new(@"^[+-]?(?<s>s)?nan(?:\d+(?:_\d+)*)?$", RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

    public DecimalKind Kind { get; }
    public bool Negative { get; }
    public BigInteger Coefficient { get; }
    public BigInteger Exponent { get; }

    private PythonDecimal(DecimalKind kind, bool negative, BigInteger coefficient, BigInteger exponent)
    {
        Kind = kind;
        Negative = negative;
        Coefficient = coefficient;
        Exponent = exponent;
    }

    public bool IsFinite => Kind == DecimalKind.Finite;

    private static BigInteger Digits(string text)
    {
        BigInteger value = BigInteger.Zero;
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == '_') continue;
            value = value * 10 + CharUnicodeInfo.GetDecimalDigitValue(text[i]);
        }
        return value;
    }

    /// <summary>Parses as <c>decimal.Decimal(text)</c> would; false where Python raises InvalidOperation.</summary>
    public static bool TryParse(string text, out PythonDecimal value)
    {
        string s = PythonText.Strip(text);
        value = default;
        var m = FiniteForm.Match(s);
        if (m.Success)
        {
            string frac = m.Groups["frac"].Success ? m.Groups["frac"].Value.Replace("_", "") : "";
            BigInteger coefficient = Digits((m.Groups["int"].Success ? m.Groups["int"].Value : "") + frac);
            BigInteger exponent = m.Groups["exp"].Success ? ParseSigned(m.Groups["exp"].Value) : BigInteger.Zero;
            value = new PythonDecimal(DecimalKind.Finite, m.Groups["sign"].Value == "-", coefficient, exponent - frac.Length);
            return true;
        }
        if (InfinityForm.IsMatch(s))
        {
            value = new PythonDecimal(DecimalKind.Infinity, s.StartsWith("-", StringComparison.Ordinal), 0, 0);
            return true;
        }
        var n = NaNForm.Match(s);
        if (n.Success)
        {
            value = new PythonDecimal(n.Groups["s"].Success ? DecimalKind.SignalingNaN : DecimalKind.QuietNaN,
                s.StartsWith("-", StringComparison.Ordinal), 0, 0);
            return true;
        }
        return false;
    }

    private static BigInteger ParseSigned(string text)
    {
        bool negative = text.StartsWith("-", StringComparison.Ordinal);
        string digits = text.TrimStart('+', '-');
        BigInteger v = Digits(digits);
        return negative ? -v : v;
    }

    /// <summary>True when the value has a non-zero fractional part (<c>d % 1 != 0</c>).</summary>
    public bool HasFraction
    {
        get
        {
            if (Exponent >= 0 || Coefficient.IsZero) return false;
            if (-Exponent > 100000) return true;
            return !(Coefficient % BigInteger.Pow(10, (int)(-Exponent))).IsZero;
        }
    }

    /// <summary>
    /// <c>int(d)</c>: the integer part. An exponent past <paramref name="limitDigits"/> digits throws, as the integer
    /// would exceed any number-to-words range (Python reaches the same OverflowError after building it).
    /// </summary>
    public BigInteger Truncate(int limitDigits = 10000)
    {
        BigInteger magnitude;
        if (Exponent >= 0)
        {
            if (Exponent > limitDigits)
                throw new OverflowException("The number is too large to verbalize.");
            magnitude = Coefficient * BigInteger.Pow(10, (int)Exponent);
        }
        else
        {
            magnitude = -Exponent > 100000 ? BigInteger.Zero : BigInteger.Divide(Coefficient, BigInteger.Pow(10, (int)(-Exponent)));
        }
        return Negative ? -magnitude : magnitude;
    }

    /// <summary>Whether <c>low &lt; d &lt; high</c> holds (finite values only).</summary>
    public bool Between(BigInteger low, BigInteger high)
    {
        // Past 10^10000 the value is zero or far outside any bound this is asked about.
        if (Exponent > 10000) return Coefficient.IsZero && low < 0 && high > 0;
        // Compare d * 10^k with the bounds scaled the same way, exactly.
        BigInteger scale = Exponent < 0 ? BigInteger.Pow(10, (int)BigInteger.Min(-Exponent, 100000)) : BigInteger.One;
        BigInteger scaled = Exponent < 0 ? Coefficient : Coefficient * BigInteger.Pow(10, (int)Exponent);
        if (Negative) scaled = -scaled;
        return scaled > low * scale && scaled < high * scale;
    }

    /// <summary><c>float(d)</c>: the nearest double.</summary>
    public double ToDouble()
    {
        if (Kind == DecimalKind.Infinity) return Negative ? double.NegativeInfinity : double.PositiveInfinity;
        if (Kind != DecimalKind.Finite) return double.NaN;
        string text = (Negative ? "-" : "") + Coefficient.ToString(CultureInfo.InvariantCulture) + "E" + Exponent.ToString(CultureInfo.InvariantCulture);
        try
        {
            return double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
        }
        catch (OverflowException)
        {
            // .NET Framework throws where Python (and .NET Core) round to infinity.
            return Negative ? double.NegativeInfinity : double.PositiveInfinity;
        }
    }
}
