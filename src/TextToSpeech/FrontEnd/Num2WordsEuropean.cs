using System.Globalization;
using System.Linq;
using System.Numerics;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// num2words 0.5.12 (LGPL-2.1) as gruut calls it for Spanish and French: <c>Num2Word_Base</c>/<c>Num2Word_EU</c>'s
/// cardinal machinery (the card table, <c>splitnum</c>'s nested split and <c>clean</c>'s pairwise merge), the year and
/// currency forms, and each language's <c>merge</c>. Values are arbitrary-precision, as Python's integers are.
/// </summary>
internal abstract class Num2WordsEuropean
{
    private sealed class Pair
    {
        public Pair(string text, BigInteger number)
        {
            Text = text;
            Number = number;
        }

        public string Text { get; }
        public BigInteger Number { get; }
    }

    private readonly List<BigInteger> _cardOrder = new();
    private readonly Dictionary<BigInteger, string> _cards = new();
    private readonly Dictionary<string, (string[] Units, string[] Cents)> _currencyForms;

    protected Num2WordsEuropean(IReadOnlyList<string> high, string? gigaSuffix, string? megaSuffix,
        IReadOnlyList<(int Value, string Word)> mid, IReadOnlyList<string> low, string negWord, string pointWord, string currencyKey)
    {
        NegWord = negWord;
        PointWord = pointWord;
        int cap = 3 + 6 * high.Count;
        for (int i = 0; i < high.Count; i++)
        {
            int n = cap - 6 * i;
            if (!string.IsNullOrEmpty(gigaSuffix)) SetCard(BigInteger.Pow(10, n), high[i] + gigaSuffix);
            if (!string.IsNullOrEmpty(megaSuffix)) SetCard(BigInteger.Pow(10, n - 3), high[i] + megaSuffix);
        }
        foreach (var (value, word) in mid) SetCard(value, word);
        for (int i = 0; i < low.Count; i++) SetCard(low.Count - 1 - i, low[i]);
        MaxValue = 1000 * _cardOrder[0];

        var data = CurrencyData.Value;
        _currencyForms = new Dictionary<string, (string[], string[])>(StringComparer.Ordinal);
        foreach (var entry in (JObject)data[currencyKey]!)
        {
            var forms = (JArray)entry.Value!;
            _currencyForms[entry.Key] = (((JArray)forms[0]).Select(t => (string)t!).ToArray(), ((JArray)forms[1]).Select(t => (string)t!).ToArray());
        }
    }

    internal static readonly Lazy<JObject> CurrencyData = new(
        () => JObject.Parse(string.Join("\n", ChineseText.ReadLines("num2words_currency.json.gz"))), isThreadSafe: true);

    // OrderedDict assignment: a new key goes last, an existing one keeps its place.
    private void SetCard(BigInteger key, string word)
    {
        if (!_cards.ContainsKey(key)) _cardOrder.Add(key);
        _cards[key] = word;
    }

    protected string NegWord { get; }
    protected string PointWord { get; }

    /// <summary>MAXVAL: 1000 times the largest card; cardinals at or past it raise OverflowError.</summary>
    public BigInteger MaxValue { get; }

    /// <summary>The language's <c>merge(curr, next)</c> of two (text, number) pairs.</summary>
    protected abstract (string Text, BigInteger Number) Merge(string ctext, BigInteger cnum, string ntext, BigInteger nnum);

    private List<object> SplitNum(BigInteger value)
    {
        foreach (var elem in _cardOrder)
        {
            if (elem > value) continue;
            var output = new List<object>();
            BigInteger div, mod;
            if (value.IsZero)
            {
                div = 1;
                mod = 0;
            }
            else
            {
                div = BigInteger.DivRem(value, elem, out mod);
            }
            if (div.IsOne)
            {
                output.Add(new Pair(_cards[1], 1));
            }
            else
            {
                if (div == value)
                    return new List<object> { new Pair(string.Concat(Enumerable.Repeat(_cards[elem], (int)div)), div * elem) };
                output.Add(SplitNum(div));
            }
            output.Add(new Pair(_cards[elem], elem));
            if (!mod.IsZero) output.Add(SplitNum(mod));
            return output;
        }
        throw new ArgumentOutOfRangeException(nameof(value), "No card fits the value.");
    }

    private object Clean(List<object> val)
    {
        List<object> output = val;
        while (val.Count != 1)
        {
            output = new List<object>();
            if (val[0] is Pair left && val[1] is Pair right)
            {
                var merged = Merge(left.Text, left.Number, right.Text, right.Number);
                output.Add(new Pair(merged.Text, merged.Number));
                if (val.Count > 2) output.Add(val.GetRange(2, val.Count - 2));
            }
            else
            {
                foreach (var elem in val)
                {
                    if (elem is List<object> list) output.Add(list.Count == 1 ? list[0] : Clean(list));
                    else output.Add(elem);
                }
            }
            val = output;
        }
        return output[0];
    }

    /// <summary><c>to_cardinal</c> of an integer.</summary>
    public string ToCardinal(BigInteger value)
    {
        string prefix = "";
        if (value.Sign < 0)
        {
            value = -value;
            prefix = NegWord;
        }
        if (value >= MaxValue)
            throw new OverflowException($"abs({value}) must be less than {MaxValue}.");
        var pair = (Pair)Clean(SplitNum(value));
        return prefix + pair.Text;
    }

    /// <summary><c>to_cardinal</c> of a float with a fractional part (<c>to_cardinal_float</c>).</summary>
    public string ToCardinal(double value)
    {
        if (!double.IsNaN(value) && !double.IsInfinity(value) && Math.Floor(value) == value)
            return ToCardinal(new BigInteger(value));
        var (pre, post, precision) = Float2Tuple(value);
        string postText = post.ToString(CultureInfo.InvariantCulture);
        if (precision > postText.Length) postText = new string('0', precision - postText.Length) + postText;
        var output = new List<string> { ToCardinal(pre) };
        if (precision != 0) output.Add(PointWord);
        for (int i = 0; i < precision; i++) output.Add(ToCardinal(new BigInteger(postText[i] - '0')));
        return string.Join(" ", output);
    }

    // float2tuple: the integer part, and the fraction scaled by 10^precision, precision being the digits after the
    // point in repr(value).
    private static (BigInteger Pre, BigInteger Post, int Precision) Float2Tuple(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            throw new OverflowException("cannot convert float NaN or infinity to integer");
        var pre = new BigInteger(Math.Truncate(value));
        var (digits, decimalPoint) = ShortestDigits(value);
        int precision = Math.Abs(decimalPoint - digits.Length);
        double post = Math.Abs(value - (double)pre) * Math.Pow(10, precision);
        double rounded = Math.Round(post, MidpointRounding.ToEven);
        BigInteger scaled = Math.Abs(rounded - post) < 0.01 ? new BigInteger(rounded) : new BigInteger(Math.Floor(post));
        return (pre, scaled, precision);
    }

    // repr(float)'s shortest round-tripping digits and the position of the decimal point relative to them.
    private static (string Digits, int DecimalPoint) ShortestDigits(double value)
    {
        double magnitude = Math.Abs(value);
        for (int n = 1; n <= 17; n++)
        {
            string s = magnitude.ToString("E" + (n - 1), CultureInfo.InvariantCulture);
            if (double.Parse(s, CultureInfo.InvariantCulture) != magnitude) continue;
            int e = s.IndexOf('E');
            string mantissa = s.Substring(0, e).Replace(".", "");
            int exponent = int.Parse(s.Substring(e + 1), CultureInfo.InvariantCulture);
            return (mantissa.TrimEnd('0').Length == 0 ? "0" : mantissa.TrimEnd('0'), exponent + 1);
        }
        throw new InvalidOperationException("No round-tripping representation.");
    }

    /// <summary><c>to_year</c>: the cardinal, for both languages.</summary>
    public string ToYear(BigInteger value) => ToCardinal(value);

    private static string Pluralize(BigInteger n, string[] forms) => n.IsOne ? forms[0] : forms[1];

    /// <summary>
    /// <c>to_currency(val, currency, cents=True, separator)</c> of a float: the amount rounded half-up to cents, the
    /// units and cents each as a cardinal with its currency word.
    /// </summary>
    public string ToCurrency(double value, string currency, string separator)
    {
        var (units, cents, negative) = ParseCurrencyParts(value);
        if (!_currencyForms.TryGetValue(currency, out var forms))
            throw new NotSupportedException($"Currency code \"{currency}\" not implemented for {GetType().Name}");
        string minus = negative ? NegWord + " " : "";
        string result = $"{minus}{ToCardinal(units)} {Pluralize(units, forms.Units)}{separator} {ToCardinal(cents)} {Pluralize(cents, forms.Cents)}";
        return FinishCurrency(result, currency, separator);
    }

    protected virtual string FinishCurrency(string result, string currency, string separator) => result;

    // parse_currency_parts of a float: Decimal(value) (exact), quantized to 0.01 rounding half up, split at the point.
    private static (BigInteger Units, BigInteger Cents, bool Negative) ParseCurrencyParts(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentException("A NaN or infinite amount cannot be quantized.", nameof(value));
        long bits = BitConverter.DoubleToInt64Bits(value);
        bool negative = bits < 0;
        int exponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0) exponent = 1;
        else mantissa |= 1L << 52;
        exponent -= 1075;
        BigInteger hundredths;
        if (exponent >= 0)
        {
            hundredths = new BigInteger(mantissa) * BigInteger.Pow(2, exponent) * 100;
        }
        else
        {
            BigInteger numerator = new BigInteger(mantissa) * 100;
            BigInteger denominator = BigInteger.Pow(2, -exponent);
            hundredths = BigInteger.DivRem(numerator, denominator, out var remainder);
            if (remainder * 2 >= denominator) hundredths += 1;
        }
        return (hundredths / 100, hundredths % 100, negative && !hundredths.IsZero);
    }
}

/// <summary>num2words' <c>Num2Word_ES</c>.</summary>
internal sealed class Num2WordsSpanish : Num2WordsEuropean
{
    private static readonly Lazy<Num2WordsSpanish> SharedInstance = new(() => new Num2WordsSpanish(), isThreadSafe: true);
    private readonly HashSet<string> _currenciesUna;
    private readonly HashSet<string> _centsUna;

    public static Num2WordsSpanish Default => SharedInstance.Value;

    private Num2WordsSpanish()
        : base(new[] { "cuatr", "tr", "b", "m" }, null, "illón",
            new[] { (1000, "mil"), (100, "cien"), (90, "noventa"), (80, "ochenta"), (70, "setenta"), (60, "sesenta"),
                (50, "cincuenta"), (40, "cuarenta"), (30, "treinta") },
            new[] { "veintinueve", "veintiocho", "veintisiete", "veintiséis", "veinticinco", "veinticuatro", "veintitrés",
                "veintidós", "veintiuno", "veinte", "diecinueve", "dieciocho", "diecisiete", "dieciséis", "quince",
                "catorce", "trece", "doce", "once", "diez", "nueve", "ocho", "siete", "seis", "cinco", "cuatro", "tres",
                "dos", "uno", "cero" },
            "menos ", "punto", "es")
    {
        _currenciesUna = new HashSet<string>(((JArray)CurrencyData.Value["es_currencies_una"]!).Select(t => (string)t!), StringComparer.Ordinal);
        _centsUna = new HashSet<string>(((JArray)CurrencyData.Value["es_cents_una"]!).Select(t => (string)t!), StringComparer.Ordinal);
    }

    private const string GenderStem = "o";

    protected override (string Text, BigInteger Number) Merge(string ctext, BigInteger cnum, string ntext, BigInteger nnum)
    {
        if (cnum.IsOne)
        {
            if (nnum < 1000000) return (ntext, nnum);
            ctext = "un";
        }
        else if (cnum == 100 && !(nnum % 1000).IsZero)
        {
            ctext += "t" + GenderStem;
        }
        if (nnum < cnum)
        {
            if (cnum < 100) return ($"{ctext} y {ntext}", cnum + nnum);
            return ($"{ctext} {ntext}", cnum + nnum);
        }
        else if ((nnum % 1000000).IsZero && cnum > 1)
        {
            ntext = ntext.Substring(0, ntext.Length - 3) + "lones";
        }
        if (nnum == 100)
        {
            if (cnum == 5)
            {
                ctext = "quinien";
                ntext = "";
            }
            else if (cnum == 7) ctext = "sete";
            else if (cnum == 9) ctext = "nove";
            ntext += "t" + GenderStem + "s";
        }
        else
        {
            ntext = " " + ntext;
        }
        return (ctext + ntext, cnum * nnum);
    }

    protected override string FinishCurrency(string result, string currency, string separator)
    {
        var parts = result.Split(new[] { separator + " " }, StringSplitOptions.None);
        if (_currenciesUna.Contains(currency))
        {
            parts[0] = parts[0].Replace("uno", "una");
            parts[0] = parts[0].Replace("cientos", "cientas");
        }
        parts[0] = parts[0].Replace("veintiuno", "veintiún");
        parts[0] = parts[0].Replace("uno", "un");
        if (_centsUna.Contains(currency)) parts[1] = parts[1].Replace("uno", "una");
        parts[1] = parts[1].Replace("veintiuno", "veintiún");
        parts[1] = parts[1].Replace("uno", "un");
        return string.Join(separator + " ", parts);
    }
}

/// <summary>num2words' <c>Num2Word_FR</c>.</summary>
internal sealed class Num2WordsFrench : Num2WordsEuropean
{
    private static readonly Lazy<Num2WordsFrench> SharedInstance = new(() => new Num2WordsFrench(), isThreadSafe: true);

    public static Num2WordsFrench Default => SharedInstance.Value;

    private static IReadOnlyList<string> HighNumwords()
    {
        string[] lows = { "non", "oct", "sept", "sext", "quint", "quadr", "tr", "b", "m" };
        string[] units = { "", "un", "duo", "tre", "quattuor", "quin", "sex", "sept", "octo", "novem" };
        string[] tens = { "dec", "vigint", "trigint", "quadragint", "quinquagint", "sexagint", "septuagint", "octogint", "nonagint" };
        var generated = new List<string>();
        foreach (var t in tens)
            foreach (var u in units) generated.Add(u + t);
        generated.Reverse();
        var high = new List<string> { "cent" };
        high.AddRange(generated);
        high.AddRange(lows);
        return high;
    }

    private Num2WordsFrench()
        : base(HighNumwords(), "illiard", "illion",
            new[] { (1000, "mille"), (100, "cent"), (80, "quatre-vingts"), (60, "soixante"), (50, "cinquante"),
                (40, "quarante"), (30, "trente") },
            new[] { "vingt", "dix-neuf", "dix-huit", "dix-sept", "seize", "quinze", "quatorze", "treize", "douze",
                "onze", "dix", "neuf", "huit", "sept", "six", "cinq", "quatre", "trois", "deux", "un", "zéro" },
            "moins ", "virgule", "fr")
    {
    }

    private static BigInteger PyMod(BigInteger a, BigInteger b)
    {
        var r = a % b;
        return r.Sign < 0 ? r + b : r;
    }

    protected override (string Text, BigInteger Number) Merge(string ctext, BigInteger cnum, string ntext, BigInteger nnum)
    {
        if (cnum.IsOne)
        {
            if (nnum < 1000000) return (ntext, nnum);
        }
        else
        {
            if ((PyMod(cnum - 80, 100).IsZero || (PyMod(cnum, 100).IsZero && cnum < 1000)) && nnum < 1000000
                && ctext[ctext.Length - 1] == 's')
                ctext = ctext.Substring(0, ctext.Length - 1);
            if (cnum < 1000 && nnum != 1000 && ntext[ntext.Length - 1] != 's' && PyMod(nnum, 100).IsZero)
                ntext += "s";
        }
        if (nnum < cnum && cnum < 100)
        {
            if (PyMod(nnum, 10).IsOne && cnum != 80) return ($"{ctext} et {ntext}", cnum + nnum);
            return ($"{ctext}-{ntext}", cnum + nnum);
        }
        if (nnum > cnum) return ($"{ctext} {ntext}", cnum * nnum);
        return ($"{ctext} {ntext}", cnum + nnum);
    }

    /// <summary><c>to_ordinal</c> (French dates read the day as an ordinal).</summary>
    public string ToOrdinal(BigInteger value)
    {
        if (value.Sign < 0) throw new ArgumentException($"Cannot treat negative num {value} as ordinal.", nameof(value));
        if (value.IsOne) return "premier";
        string word = ToCardinal(value);
        foreach (var (source, replacement) in new[] { ("cinq", "cinquième"), ("neuf", "neuvième") })
            if (word.EndsWith(source, StringComparison.Ordinal))
                return word.Substring(0, word.Length - source.Length) + replacement;
        if (word[word.Length - 1] == 'e') word = word.Substring(0, word.Length - 1);
        return word + "ième";
    }
}
