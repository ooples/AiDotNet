using System.Numerics;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's Arabic grouping, gender and currency rules. One instance is scoped to one utterance.</summary>
internal sealed class Num2WordsArabic : Num2WordsBase
{
    private static readonly Lazy<JObject> Data = new(() => JObject.Parse(string.Join("\n", ChineseText.ReadLines("num2words_arabic.json.gz"))));
    private bool _feminine;
    private static readonly string[] Empty = ["", "", "", ""];
    internal Num2WordsArabic() : base(Enumerable.Range(0, 10).Reverse().Select(i => (new BigInteger(i), i.ToString())), "سالب ", ",", " ", null) { }
    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
        => throw new InvalidOperationException("Arabic uses its gender-sensitive grouping converter instead of card merging.");
    private static string At(string table, int index) => (string)Data.Value[table]![index]!;
    private string Digit(int value, int level) => At(level == -1 || (level == 0 && _feminine) ? "arabicFeminineOnes" : "arabicOnes", value);

    private string Group(int number, int level, BigInteger remaining, BigInteger integer)
    {
        int tens = number % 100, hundreds = number / 100;
        string result = hundreds > 0 ? tens == 0 && hundreds == 2 ? At("arabicAppendedTwos", 0) : At("arabicHundreds", hundreds) : "";
        if (tens == 0) return result;
        if (tens < 20)
        {
            if (tens == 2 && hundreds == 0 && level > 0)
            {
                bool appended = Enumerable.Range(1, 6).Any(i => integer == 2 * BigInteger.Pow(1000, i));
                result = At(appended ? "arabicAppendedTwos" : "arabicTwos", level);
            }
            else
            {
                if (result.Length > 0) result += " و ";
                // The source compares the untruncated Decimal(number)/100 with zero in these branches.
                if (!(tens == 1 && level > 0 && number == 0)
                    && !((tens == 1 || tens == 2) && (level == 0 || level == -1) && number == 0 && remaining.IsZero))
                    result += Digit(tens, level);
            }
        }
        else
        {
            int one = tens % 10;
            if (one > 0)
            {
                if (result.Length > 0 && tens < 60) result += " و ";
                result += Digit(one, level);
            }
            if (result.Length > 0 && one != 0) result += " و ";
            result += At("arabicTens", tens / 10 - 2);
        }
        return result;
    }

    // Python's format(value, '.9f') rounds the exact binary float to nine places, even for integers larger than 2^53.
    private static (BigInteger Integer, int Decimal) FixedNine(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0) throw new ArgumentOutOfRangeException(nameof(value));
        long bits = BitConverter.DoubleToInt64Bits(value);
        int exponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0) exponent = 1; else mantissa |= 1L << 52;
        exponent -= 1075;
        BigInteger scaled = new BigInteger(mantissa) * 1000000000;
        if (exponent >= 0) scaled <<= exponent;
        else
        {
            BigInteger denominator = BigInteger.One << -exponent;
            BigInteger quotient = BigInteger.DivRem(scaled, denominator, out var remainder);
            int comparison = (remainder * 2).CompareTo(denominator);
            scaled = quotient + (comparison > 0 || (comparison == 0 && !quotient.IsEven) ? 1 : 0);
        }
        return (scaled / 1000000000, (int)(scaled % 1000000000 / 10000000));
    }

    private string Convert(double value, string[] units, string[] subunits, string separator)
    {
        if (value == 0) return "صفر";
        var (integer, decimals) = FixedNine(value);
        string decimalText = Group(decimals, -1, BigInteger.Zero, integer);
        BigInteger remainder = integer;
        string result = ""; int level = 0;
        while (remainder > 0)
        {
            int chunk = (int)(remainder % 1000); remainder /= 1000;
            string description = Group(chunk, level, remainder, integer);
            if (description.Length > 0)
            {
                if (level > 0)
                {
                    if (result.Length > 0) result = " و " + result;
                    if (chunk != 2)
                    {
                        string table = chunk % 100 == 1 ? "arabicGroup" : chunk is >= 3 and <= 10 ? "arabicPluralGroups"
                            : result.Length > 0 ? "arabicAppendedGroup" : "arabicGroup";
                        result = At(table, level) + " " + result;
                    }
                }
                result = description + " " + result;
            }
            level++;
        }
        static int Form(int value, bool two) => value == 2 && two ? 1 : value >= 3 && value <= 10 ? 2 : value >= 11 ? 3 : 0;
        if (!integer.IsZero) result += units[Form((int)(integer % 100), integer == 2)];
        if (decimals != 0) result += " " + separator + " " + decimalText + " " + subunits[Form(decimals % 100, true)];
        return result;
    }

    public override string ToCardinal(BigInteger value)
    {
        if (value >= BigInteger.Pow(10, 36)) throw new OverflowException("Arabic number exceeds num2words' limit.");
        return (value.Sign < 0 ? "سالب " : "") + Convert((double)BigInteger.Abs(value), Empty, Empty, ",").Trim();
    }
    public override string ToCardinal(double value)
    {
        if (value >= 1e36) throw new OverflowException("Arabic number exceeds num2words' limit.");
        return (value < 0 ? "سالب " : "") + Convert(Math.Abs(value), Empty, Empty, ",").Trim();
    }
    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        if (value <= 19) return At("arabicOrdinal", (int)value);
        _feminine = value < 100;
        return Convert((double)value, Empty, Empty, "و").Trim();
    }
    public override string ToCurrency(double value, string currency, string separator)
    {
        _feminine = false;
        string key = currency is "EGP" or "KWD" ? currency : "SR";
        var forms = Data.Value["currencies"]![key]!;
        return Convert(value, ((JArray)forms[0]!).Select(item => (string)item!).ToArray(),
            ((JArray)forms[1]!).Select(item => (string)item!).ToArray(), "و");
    }
}
