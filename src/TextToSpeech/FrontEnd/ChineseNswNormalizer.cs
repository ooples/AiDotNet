using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// chn_text_norm's non-standard-word normalizer (Zhiyang Zhou and Jiayu Du, 2019; <c>utils/text_norm.py</c> in DiffSinger
/// and the authors' other NATSpeech-family code bases, <c>NSWNormalizer</c>): dates, money, mobile and fixed telephone
/// numbers, fractions, percentages, numbers before a measure word, long digit strings and plain numbers rewritten in
/// Chinese characters, in that order, each pattern's matches found once and replaced first-occurrence by
/// first-occurrence.
/// </summary>
internal static class ChineseNswNormalizer
{
    private const string Digits = "零一二三四五六七八九";
    private const string BigSimplified = "零壹贰叁肆伍陆柒捌玖";
    private const string BigTraditional = "零壹貳參肆伍陸柒捌玖";

    private const string CurrencyUnits = "((亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|)元|(亿|千万|百万|万|千|百|)块|角|毛|分)";
    private const string Quantifiers = "(匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|"
        + "砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|"
        + "针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|"
        + "毫|厘|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|"
        + "盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|旬|"
        + "纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块";

    private static readonly Regex DatePattern = new(@"\D+((([089]\d|(19|20)\d{2})年)?(\d{1,2}月(\d{1,2}[日号])?)?)");
    private static readonly Regex MoneyPattern = new(@"\D+((\d+(\.\d+)?)[多余几]?" + CurrencyUnits + @"(\d" + CurrencyUnits + "?)?)");
    private static readonly Regex MobilePattern = new(@"\D((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})\D");
    private static readonly Regex FixedPattern = new(@"\D((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})\D");
    private static readonly Regex FractionPattern = new(@"(\d+/\d+)");
    private static readonly Regex PercentagePattern = new(@"(\d+(\.\d+)?%)");
    private static readonly Regex QuantifierPattern = new(@"(\d+(\.\d+)?)[多余几]?" + Quantifiers + ")");
    // Fish Speech 1.4's copy (fish_speech/text/chn_text_norm) adds 人 and 抽 to the measure words.
    private static readonly Regex FishSpeechQuantifierPattern = new(@"(\d+(\.\d+)?)[多余几]?" + Quantifiers + "|人|抽)");
    private static readonly Regex DigitStringPattern = new(@"(\d{4,32})");
    private static readonly Regex NumberPattern = new(@"(\d+(\.\d+)?)");
    private static readonly Regex Particular = new("(([a-zA-Z]+)二([a-zA-Z]+))");

    /// <summary><c>NSWNormalizer(text).normalize(remove_punc=False)</c>; with <paramref name="fishSpeech"/>, Fish Speech
    /// 1.4's copy (<c>ChnNormedText(raw_text=text).normalize()</c>), whose measure words add 人 and 抽.</summary>
    public static string Normalize(string raw, bool fishSpeech = false)
    {
        string text = "^" + raw + "$";
        text = ReplaceEach(text, DatePattern, g => g[1], DateToChinese);
        text = ReplaceEach(text, MoneyPattern, g => g[1], MoneyToChinese);
        text = ReplaceEach(text, MobilePattern, g => g[1], t => Telephone(t, fixedLine: false));
        text = ReplaceEach(text, FixedPattern, g => g[1], t => Telephone(t, fixedLine: true));
        text = ReplaceEach(text, FractionPattern, g => g[1], f =>
        {
            var parts = f.Split('/');
            return NumberToChinese(parts[1]) + "分之" + NumberToChinese(parts[0]);
        });
        text = text.Replace('％', '%');
        text = ReplaceEach(text, PercentagePattern, g => g[1], p => "百分之" + NumberToChinese(p.Trim().Trim('%')));
        text = ReplaceEach(text, fishSpeech ? FishSpeechQuantifierPattern : QuantifierPattern, g => g[1], n => NumberToChinese(n));
        text = ReplaceEach(text, DigitStringPattern, g => g[1], d => NumberToChinese(d, alternateTwo: false, useUnits: false));
        text = ReplaceEach(text, NumberPattern, g => g[1], n => NumberToChinese(n));
        foreach (Match m in Particular.Matches(text).Cast<Match>().ToList())
            text = ReplaceFirst(text, m.Groups[1].Value, m.Groups[2].Value + "2" + m.Groups[3].Value);
        return text.TrimStart('^').TrimEnd('$');
    }

    // re.findall, then str.replace(found, rewritten, 1) for each, in order.
    private static string ReplaceEach(string text, Regex pattern, Func<GroupCollection, Group> pick, Func<string, string> rewrite)
    {
        foreach (var found in pattern.Matches(text).Cast<Match>().Select(m => pick(m.Groups).Value).ToList())
            text = ReplaceFirst(text, found, rewrite(found));
        return text;
    }

    // Python's str.replace(old, new, 1), including the empty pattern (which inserts at the start).
    private static string ReplaceFirst(string text, string old, string replacement)
    {
        int index = text.IndexOf(old, StringComparison.Ordinal);
        return index < 0 ? text : text.Substring(0, index) + replacement + text.Substring(index + old.Length);
    }

    // Date.date2chntext.
    private static string DateToChinese(string date)
    {
        string year, other;
        int y = date.Trim().IndexOf('年');
        if (y >= 0)
        {
            string trimmed = date.Trim();
            year = NumberToChinese(trimmed.Substring(0, y), alternateTwo: false, useUnits: false) + "年";
            other = trimmed.Substring(y + 1);
        }
        else
        {
            other = date;
            year = "";
        }
        string month = "", day = "";
        if (other.Length > 0)
        {
            string o = other.Trim();
            int m = o.IndexOf('月');
            if (m >= 0)
            {
                month = NumberToChinese(o.Substring(0, m)) + "月";
                day = o.Substring(m + 1);
            }
            else
            {
                day = date;
                month = "";
            }
            if (day.Length > 0) day = NumberToChinese(day.Substring(0, day.Length - 1)) + day[^1];
        }
        return year + month + day;
    }

    private static readonly Regex MoneyNumber = new(@"(\d+(\.\d+)?)");

    // Money.money2chntext: every number replaced (str.replace without a count).
    private static string MoneyToChinese(string money)
    {
        foreach (var number in MoneyNumber.Matches(money).Cast<Match>().Select(m => m.Groups[1].Value).ToList())
            money = money.Replace(number, NumberToChinese(number));
        return money;
    }

    // TelePhone.telephone2chntext.
    private static string Telephone(string telephone, bool fixedLine)
    {
        var parts = fixedLine ? telephone.Split('-')
            : telephone.Trim('+').Split(new[] { ' ', '\t', '\n', '\r', '\v', '\f' }, StringSplitOptions.RemoveEmptyEntries);
        return string.Concat(parts.Select(p => NumberToChinese(p, alternateTwo: false, useUnits: false)));
    }

    // The mid numbering system's units: 十 百 千 万 (10^1–10^4), then 亿 兆 京 垓 秭 穰 沟 涧 正 载 (10^8, 10^12, ...).
    private static readonly (int Power, char Simplified)[] Units =
        "十百千万".Select((c, i) => (i + 1, c)).Concat("亿兆京垓秭穰沟涧正载".Select((c, i) => ((i + 2) * 4, c))).ToArray();

    // A symbol of num2chn's result: a digit (value), a unit (power) or the point.
    private readonly struct Symbol
    {
        public Symbol(int kind, int value) { Kind = kind; Value = value; }
        public int Kind { get; }   // 0 digit, 1 unit, 2 point, 3 两
        public int Value { get; }
    }

    /// <summary>
    /// <c>num2chn(number_string, alt_two=..., use_units=...)</c> in the mid system, simplified, not big: digits with units
    /// (leading zeros read as 零), a decimal point and digits, 二 before a unit above 十 as 两 (unless after 十), a leading
    /// 点 prefixed with 零, and 一十 read as 十.
    /// </summary>
    public static string NumberToChinese(string number, bool alternateTwo = true, bool useUnits = true)
    {
        var parts = number.Split('.');
        if (parts.Length > 2) throw new ArgumentException($"invalid input num string with more than one dot: {number}");
        string integer = parts[0], fraction = parts.Length == 2 ? parts[1] : "";
        List<Symbol> symbols;
        if (useUnits && integer.Length > 1) symbols = Value(integer);
        else symbols = integer.Select(c => new Symbol(0, DigitValue(c))).ToList();
        if (fraction.Length > 0)
        {
            symbols.Add(new Symbol(2, 0));
            symbols.AddRange(fraction.Select(c => new Symbol(0, DigitValue(c))));
        }
        if (alternateTwo)
            for (int i = 0; i < symbols.Count; i++)
            {
                if (symbols[i].Kind != 0 || symbols[i].Value != 2) continue;
                Symbol? next = i < symbols.Count - 1 ? symbols[i + 1] : null;
                Symbol? previous = i > 0 ? symbols[i - 1] : null;
                if (next is { Kind: 1 } n && (previous is null || previous.Value.Kind == 1))
                    if (n.Value != 1 && (previous is null || previous.Value.Value != 1)) symbols[i] = new Symbol(3, 2);
            }
        var result = new StringBuilder();
        foreach (var s in symbols)
            result.Append(s.Kind switch
            {
                0 => Digits[s.Value],
                1 => Units.First(u => u.Power == s.Value).Simplified,
                2 => '点',
                _ => '两',
            });
        string text = result.ToString();
        if (text.StartsWith("点", StringComparison.Ordinal)) return Digits[0] + text;
        if (text.Length >= 2 && (text[1] == '十' || text[1] == '拾') && (text[0] == Digits[1] || text[0] == BigSimplified[1] || text[0] == BigTraditional[1]))
            text = text.Substring(1);
        return text;
    }

    private static int DigitValue(char c) => (int)char.GetNumericValue(c) is int v && v >= 0 && v <= 9
        ? v : throw new FormatException($"invalid literal for int() with base 10: '{c}'");

    // get_value: the largest unit below the length, the part above it (with its leading zeros) and the stripped part
    // below it, recursively.
    private static List<Symbol> Value(string value)
    {
        string stripped = value.TrimStart('0');
        if (stripped.Length == 0) return new List<Symbol>();
        if (stripped.Length == 1)
            return value.Length != stripped.Length
                ? new List<Symbol> { new(0, 0), new(0, DigitValue(stripped[0])) }
                : new List<Symbol> { new(0, DigitValue(stripped[0])) };
        var unit = Enumerable.Reverse(Units).First(u => u.Power < stripped.Length);
        string upper = value.Substring(0, value.Length - unit.Power);
        var result = Value(upper);
        result.Add(new Symbol(1, unit.Power));
        result.AddRange(Value(stripped.Substring(stripped.Length - unit.Power)));
        return result;
    }
}
