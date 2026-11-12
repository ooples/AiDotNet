using System.Numerics;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's Turkish concatenated triplets, including the release's conditional omission branches.</summary>
internal sealed class Num2WordsTurkish : Num2WordsBase
{
    private static readonly string[] Ones = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
    private static readonly string[] Tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"];
    private static readonly string[] OrdinalOnes = ["", "birinci", "ikinci", "üçüncü", "dördüncü", "beşinci", "altıncı", "yedinci", "sekizinci", "dokuzuncu"];
    private static readonly string[] OrdinalTens = ["", "onuncu", "yirminci", "otuzuncu", "kırkıncı", "ellinci", "altmışıncı", "yetmişinci", "sekseninci", "doksanıncı"];
    private static readonly string[] Groups = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon", "kentilyon"];
    private static readonly string[] OrdinalGroups = ["", "bininci", "milyonuncu", "milyarıncı", "trilyonuncu", "katrilyonuncu", "kentilyon"];
    private static readonly BigInteger Maximum = BigInteger.Pow(10, 21) - 1;
    private static readonly Lazy<Num2WordsTurkish> Shared = new(() => new Num2WordsTurkish());
    internal static Num2WordsTurkish Default => Shared.Value;
    private Num2WordsTurkish() : base(Enumerable.Range(0, 10).Reverse().Select(i => (new BigInteger(i), Ones[i])), "eksi", "virgül", "", null) { }
    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
        => throw new InvalidOperationException("Turkish uses triplet composition instead of card merging.");

    private static string Triplet(int value, bool ordinal)
    {
        if (value == 0) return ordinal ? "sıfırıncı" : "sıfır";
        int hundred = value / 100, ten = value / 10 % 10, one = value % 10;
        string result = hundred > 0 ? (hundred == 1 ? "" : Ones[hundred]) + "yüz" : "";
        if (ordinal && ten == 0 && one == 0) return (hundred == 1 ? "" : Ones[hundred]) + "yüzüncü";
        if (ten > 0) result += ordinal && one == 0 ? OrdinalTens[ten] : Tens[ten];
        if (one > 0) result += ordinal ? OrdinalOnes[one] : Ones[one];
        return result;
    }

    private static string Read(BigInteger number, bool ordinal)
    {
        if (number.Sign < 0) throw new ArgumentOutOfRangeException(nameof(number));
        if (number >= Maximum) throw new OverflowException("Turkish number exceeds num2words' maximum.");
        // verify_cardinal compares Python float(value) with the original integer, and returns an empty string on mismatch.
        if (!ordinal && new BigInteger((double)number) != number) return "";
        if (number.IsZero) return Triplet(0, ordinal);
        var chunks = new List<int>();
        while (number > 0) { chunks.Add((int)(number % 1000)); number /= 1000; }
        int last = chunks.FindIndex(value => value != 0);
        var result = new StringBuilder();
        for (int level = chunks.Count - 1; level >= 0; level--)
        {
            int value = chunks[level]; if (value == 0) continue;
            if (level == 0) { result.Append(Triplet(value, ordinal)); continue; }
            if (value != 1 || level != 1)
            {
                string prefix = Triplet(value, false);
                // Released leading XYZ,000-with-remainder branch omits its last bir in the thousands coefficient.
                if (level == 1 && chunks.Count == 2 && value >= 100 && value % 10 == 1 && chunks[0] > 0)
                    prefix = prefix.Substring(0, prefix.Length - 3);
                // Released intermediate 00X groups differ between the cardinal and ordinal branches.
                if (level < chunks.Count - 1 && level > 1 && level > last && value < 10 && (!ordinal || value == 1)) prefix = "";
                result.Append(prefix);
            }
            result.Append(ordinal && level == last ? OrdinalGroups[level] : Groups[level]);
        }
        return result.ToString();
    }

    public override string ToCardinal(BigInteger value) => Read(value, false);
    internal string ToOrdinal(BigInteger value) => Read(value, true);
    public override string ToCardinal(double value)
    {
        if (value < 0 || double.IsNaN(value) || double.IsInfinity(value)) throw new ArgumentOutOfRangeException(nameof(value));
        if (value == Math.Truncate(value)) return Read(new BigInteger(value), false);
        BigInteger hundredths = new(Math.Truncate(value * 100));
        int fraction = (int)(hundredths % 100);
        var whole = new BigInteger(Math.Truncate(value));
        return (whole.IsZero ? "sıfır" : Read(whole, false)) + "virgül" + Tens[fraction / 10] + Ones[fraction % 10];
    }
    public override string ToCurrency(double value, string currency, string separator)
        => throw new NotSupportedException("The released Turkish num2words currency method does not accept a currency code; XTTS falls through to ordinary number expansion.");
}
