using System.Numerics;

namespace AiDotNet.TextToSpeech.FrontEnd;

internal sealed partial class Num2WordsSpanish
{
    private static readonly IReadOnlyDictionary<BigInteger, string> OrdinalStems = new Dictionary<BigInteger, string>
    {
        [1] = "primer", [2] = "segund", [3] = "tercer", [4] = "cuart", [5] = "quint", [6] = "sext",
        [7] = "séptim", [8] = "octav", [9] = "noven", [10] = "décim", [20] = "vigésim", [30] = "trigésim",
        [40] = "quadragésim", [50] = "quincuagésim", [60] = "sexagésim", [70] = "septuagésim", [80] = "octogésim",
        [90] = "nonagésim", [100] = "centésim", [200] = "ducentésim", [300] = "tricentésim", [400] = "cuadrigentésim",
        [500] = "quingentésim", [600] = "sexcentésim", [700] = "septigentésim", [800] = "octigentésim", [900] = "noningentésim",
        [1000] = "milésim", [1000000] = "millonésim", [1000000000] = "billonésim",
        [BigInteger.Pow(10, 12)] = "trillonésim", [BigInteger.Pow(10, 15)] = "cuadrillonésim"
    };

    /// <summary>num2words 0.5.12's masculine ordinal form, including its published spellings.</summary>
    internal string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        if (value.IsZero) return string.Empty;
        if (value <= 10) return OrdinalStems[value] + "o";
        if (value <= 12) return OrdinalStems[10] + "o" + ToOrdinal(value - 10);
        if (value <= 100)
        {
            var tens = value / 10 * 10;
            return (OrdinalStems[tens] + "o " + ToOrdinal(value - tens)).Trim();
        }
        if (value <= 1000)
        {
            var hundreds = value / 100 * 100;
            return (OrdinalStems[hundreds] + "o " + ToOrdinal(value - hundreds)).Trim();
        }
        if (value < BigInteger.Pow(10, 18))
        {
            BigInteger scale = 1000;
            while (scale * 1000 <= value) scale *= 1000;
            var high = value / scale;
            return ((high.IsOne ? "" : ToCardinal(high)) + OrdinalStems[scale] + "o " + ToOrdinal(value % scale)).Trim();
        }
        return ToCardinal(value);
    }
}
