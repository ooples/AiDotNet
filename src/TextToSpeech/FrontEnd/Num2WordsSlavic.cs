using System.Globalization;
using System.Numerics;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>num2words 0.5.12's Russian, Polish and Czech three-digit grouping and inflection rules.</summary>
internal sealed class Num2WordsSlavic : Num2WordsBase
{
    private static readonly Lazy<JObject> Tables = new(() => JObject.Parse(string.Join("\n", ChineseText.ReadLines("num2words_slavic.json.gz"))));
    private static readonly Lazy<Num2WordsSlavic> Ru = new(() => new Num2WordsSlavic("ru"));
    private static readonly Lazy<Num2WordsSlavic> Pl = new(() => new Num2WordsSlavic("pl"));
    private static readonly Lazy<Num2WordsSlavic> Cz = new(() => new Num2WordsSlavic("cz"));
    internal static Num2WordsSlavic Russian => Ru.Value;
    internal static Num2WordsSlavic Polish => Pl.Value;
    internal static Num2WordsSlavic Czech => Cz.Value;
    private readonly string _language;
    private JToken Data => Tables.Value[_language]!;
    private Num2WordsSlavic(string language) : base(Enumerable.Range(0, 20).Reverse().Select(i => (new BigInteger(i), i.ToString(CultureInfo.InvariantCulture))),
        (string)Tables.Value[language]!["negative"]!, (string)Tables.Value[language]!["point"]!, " ", language) => _language = language;
    protected override (string Text, BigInteger Number) Merge(string left, BigInteger l, string right, BigInteger r)
        => throw new InvalidOperationException("Slavic converters use three-digit grouping instead of card merging.");
    private string Word(string table, int index, int form = 0) => (string)Data[table]![index.ToString(CultureInfo.InvariantCulture)]![form]!;

    protected override string Pluralize(BigInteger number, string[] forms)
    {
        if (_language == "ru")
        {
            if (number % 100 < 10 || number % 100 > 20)
            {
                if (number % 10 == 1) return forms[0];
                if (number % 10 > 1 && number % 10 < 5) return forms[1];
            }
            return forms[2];
        }
        return number.IsOne ? forms[0] : number % 10 > 1 && number % 10 < 5 && (number % 100 < 10 || number % 100 > 20)
            ? forms[1] : forms[2];
    }

    public override string ToCardinal(BigInteger value)
    {
        if (value.Sign < 0) return NegWord + " " + ToCardinal(-value);
        if (value.IsZero) return (string)Data["ZERO"]![0]!;
        var chunks = new List<int>();
        while (value > 0) { chunks.Add((int)(value % 1000)); value /= 1000; }
        var words = new List<string>();
        for (int level = chunks.Count - 1; level >= 0; level--)
        {
            int chunk = chunks[level]; if (chunk == 0) continue;
            int one = chunk % 10, ten = chunk / 10 % 10, hundred = chunk / 100;
            if (hundred > 0) words.Add(Word("HUNDREDS", hundred));
            if (ten > 1) words.Add(Word("TWENTIES", ten));
            if (ten == 1) words.Add(Word("TENS", one));
            else if (one > 0 && (_language == "ru" || !(level > 0 && chunk == 1)))
                words.Add(Word(_language == "ru" && level == 1 ? "ONES_FEMININE" : "ONES", one));
            if (level > 0)
            {
                var forms = ((JArray)Data["THOUSANDS"]![level.ToString(CultureInfo.InvariantCulture)]!).Select(token => (string)token!).ToArray();
                words.Add(Pluralize(chunk, forms));
            }
        }
        return string.Join(" ", words);
    }

    public override string ToCardinal(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || (value != 0 && (Math.Abs(value) < .0001 || Math.Abs(value) >= 1e16)))
            throw new FormatException("The original Slavic decimal reader requires a non-scientific decimal representation.");
        string text = value.ToString("R", CultureInfo.InvariantCulture);
        string fraction = text.IndexOf('.') < 0 ? "0" : text.Substring(text.IndexOf('.') + 1);
        int zeros = fraction.TakeWhile(c => c == '0').Count();
        return ToCardinal(new BigInteger(Math.Truncate(value))) + " " + PointWord + " "
            + string.Concat(Enumerable.Repeat((string)Data["ZERO"]![0]! + " ", zeros))
            + ToCardinal(BigInteger.Parse(fraction, CultureInfo.InvariantCulture));
    }

    internal string ToOrdinal(BigInteger value)
    {
        if (_language == "cz") throw new NotSupportedException("num2words 0.5.12 does not implement Czech ordinals.");
        if (value.Sign < 0) throw new ArgumentOutOfRangeException(nameof(value));
        return _language == "pl" ? PolishOrdinal(value) : RussianOrdinal(value);
    }

    private string PolishOrdinal(BigInteger value)
    {
        if (value.IsZero) throw new ArgumentOutOfRangeException(nameof(value), "The original Polish ordinal reader does not support zero.");
        int level = 0; BigInteger scale = 1;
        while (value / scale % 1000 == 0) { scale *= 1000; level++; }
        int last = (int)(value / scale % 1000), one = last % 10, ten = last / 10 % 10, hundred = last / 100;
        var words = new List<string>();
        BigInteger prefix = value - last * scale;
        if (prefix > 0) words.Add(ToCardinal(prefix));
        int form = level == 0 ? 0 : 1, lastTwo = last % 100;
        if (lastTwo == 0) words.Add(Word("HUNDREDS_ORDINALS", hundred, form));
        else if (!(form == 1 && last == 1))
        {
            if (hundred > 0) words.Add(Word("HUNDREDS", hundred, lastTwo >= 20 && one != 0 ? 0 : form));
            if (lastTwo < 20) words.Add(Word("ONES_ORDINALS", lastTwo, form));
            else if (one == 0) words.Add(Word("TWENTIES_ORDINALS", ten, form));
            else { words.Add(Word("TWENTIES_ORDINALS", ten)); words.Add(Word("ONES_ORDINALS", one)); }
        }
        string result = string.Join(" ", words);
        if (last == 1 && level > 0 && result.Length > 0) result += " ";
        return level == 0 ? result : result + (string)Data["ordinal_prefixes"]![level.ToString(CultureInfo.InvariantCulture)]!;
    }

    private string RussianOrdinal(BigInteger value)
    {
        var words = ToCardinal(value).Split(' ');
        string last = words[words.Length - 1].ToLowerInvariant();
        var feminine = (JObject)Data["feminine_ordinals"]!;
        if (words.Length > 1)
        {
            string previous = words[words.Length - 2];
            if (feminine[previous] is { } form) words[words.Length - 2] = (string)form!;
            else if (previous == "десять") words[words.Length - 2] = "десяти";
        }
        if (words.Length == 3 && words[0] is "один" or "одна") words[0] = "";
        if (Data["ordinals"]![last] is { } ordinal) last = (string)ordinal!;
        else
        {
            string stem = last.Substring(0, Math.Max(0, last.Length - 3));
            if (feminine[stem] is { } form) last = (string)form! + "сотый";
            else if (last.EndsWith("ь", StringComparison.Ordinal) || last[last.Length - 2] == 'т') last = last.Substring(0, last.Length - 1) + "ый";
            else if (last.EndsWith("к", StringComparison.Ordinal)) last += "овой";
            else if (last.EndsWith("десят", StringComparison.Ordinal)) last = last.Replace('ь', 'и') + "ый";
            else if (last[last.Length - 2] == 'ч' || last.EndsWith("ч", StringComparison.Ordinal))
            {
                if (last[last.Length - 2] == 'ч') last = last.Substring(0, last.Length - 1) + "ный";
                if (last.EndsWith("ч", StringComparison.Ordinal)) last += "ный";
            }
            else if (last.EndsWith("н", StringComparison.Ordinal) || last[last.Length - 2] == 'н') last = last.Substring(0, last.LastIndexOf('н') + 1) + "ный";
            else if (last.EndsWith("д", StringComparison.Ordinal) || last[last.Length - 2] == 'д') last = last.Substring(0, last.LastIndexOf('д') + 1) + "ный";
        }
        words[words.Length - 1] = last;
        return string.Join(" ", words).Trim();
    }
}
