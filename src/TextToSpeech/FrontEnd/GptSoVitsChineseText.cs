using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The tables GPT-SoVITS's Chinese front end reads (RVC-Boss/GPT-SoVITS at b4570b8, MIT; G2PW model 1.1, Apache-2.0),
/// exported by <c>tools/reference-data/gptsovits_chinese_resources.py</c>.
/// </summary>
internal sealed class GptSoVitsChineseData
{
    private static readonly Lazy<GptSoVitsChineseData> SharedInstance = new(() => new GptSoVitsChineseData(), isThreadSafe: true);

    public static GptSoVitsChineseData Instance => SharedInstance.Value;

    private GptSoVitsChineseData()
    {
        var data = JObject.Parse(string.Join("\n", ChineseText.ReadLines("gptsovits_chinese.json.gz")));
        foreach (var entry in (JObject)data["opencpop"]!) Opencpop[entry.Key] = (string)entry.Value!;
        foreach (var w in (JArray)data["must_neural_tone_words"]!) MustNeutral.Add((string)w!);
        foreach (var w in (JArray)data["must_not_neural_tone_words"]!) MustNotNeutral.Add((string)w!);
        foreach (var w in (JArray)data["must_erhua"]!) MustErhua.Add((string)w!);
        foreach (var w in (JArray)data["not_erhua"]!) NotErhua.Add((string)w!);
        foreach (var entry in (JObject)data["polyphonic_corrections"]!)
            Corrections[entry.Key] = ((JArray)entry.Value!).Select(v => (string)v!).ToList();
        foreach (var row in (JArray)data["g2pw_polyphonic_chars"]!) PolyphonicChars.Add(((string)row[0]!, (string)row[1]!));
        foreach (var row in (JArray)data["g2pw_monophonic_chars"]!) MonophonicChars.Add(((string)row[0]!, (string)row[1]!));
        foreach (var entry in (JObject)data["g2pw_bopomofo_to_pinyin"]!) BopomofoToPinyin[entry.Key] = (string)entry.Value!;
        foreach (var entry in (JObject)data["t2s"]!) TraditionalToSimplified[entry.Key] = (string)entry.Value!;
        SymbolsV1 = ((JArray)data["symbols_v1"]!).Select(v => (string)v!).ToList();
        SymbolsV2 = ((JArray)data["symbols_v2"]!).Select(v => (string)v!).ToList();
        foreach (var c in ChineseText.Codepoints((string)data["numeric"]!)) Numeric.Add(c);
    }

    /// <summary>opencpop-strict.txt: pinyin to its initial and final phones.</summary>
    public Dictionary<string, string> Opencpop { get; } = new(StringComparer.Ordinal);

    public HashSet<string> MustNeutral { get; } = new(StringComparer.Ordinal);

    public HashSet<string> MustNotNeutral { get; } = new(StringComparer.Ordinal);

    public HashSet<string> MustErhua { get; } = new(StringComparer.Ordinal);

    public HashSet<string> NotErhua { get; } = new(StringComparer.Ordinal);

    /// <summary>text/g2pw's polyphonic.rep then polyphonic-fix.rep: a word's (or a character's) TONE3 readings.</summary>
    public Dictionary<string, List<string>> Corrections { get; } = new(StringComparer.Ordinal);

    /// <summary>POLYPHONIC_CHARS.txt: (character, bopomofo) in file order.</summary>
    public List<(string Char, string Bopomofo)> PolyphonicChars { get; } = new();

    /// <summary>MONOPHONIC_CHARS.txt: (character, bopomofo) in file order.</summary>
    public List<(string Char, string Bopomofo)> MonophonicChars { get; } = new();

    public Dictionary<string, string> BopomofoToPinyin { get; } = new(StringComparer.Ordinal);

    /// <summary>zh_normalization's char_convert t2s_dict.</summary>
    public Dictionary<string, string> TraditionalToSimplified { get; } = new(StringComparer.Ordinal);

    /// <summary>text/symbols.py's symbols (version v1).</summary>
    public List<string> SymbolsV1 { get; }

    /// <summary>text/symbols2.py's symbols (v2 and later).</summary>
    public List<string> SymbolsV2 { get; }

    /// <summary>The characters Python's <c>str.isnumeric</c> accepts.</summary>
    public HashSet<string> Numeric { get; } = new(StringComparer.Ordinal);

    /// <summary><c>tranditional_to_simplified</c>: each character through t2s_dict.</summary>
    public string ToSimplified(string text)
    {
        var output = new StringBuilder(text.Length);
        foreach (var c in ChineseText.Codepoints(text))
            output.Append(TraditionalToSimplified.TryGetValue(c, out var s) ? s : c);
        return output.ToString();
    }

    /// <summary>Python's sort order for strings: by code point.</summary>
    public static int CompareCodepoints(string a, string b)
    {
        var x = ChineseText.Codepoints(a);
        var y = ChineseText.Codepoints(b);
        for (int i = 0; i < Math.Min(x.Count, y.Count); i++)
        {
            int c = char.ConvertToUtf32(x[i], 0).CompareTo(char.ConvertToUtf32(y[i], 0));
            if (c != 0) return c;
        }
        return x.Count.CompareTo(y.Count);
    }
}

/// <summary>
/// pypinyin 0.55.0 as GPT-SoVITS calls it: <c>pinyin(text, style=TONE3, neutral_tone_with_five=True)</c> and
/// <c>lazy_pinyin(word, neutral_tone_with_five=True, style=INITIALS / FINALS_TONE3)</c> (Han runs cut by maximum matching
/// on the phrase dictionary, each piece read as a phrase or character by character, the first reading; any other run is
/// one item, kept as written), and the <c>contrib.tone_convert</c> functions <c>to_initials</c> and
/// <c>to_finals_tone3</c>.
/// </summary>
internal sealed class GptSoVitsPinyin
{
    private static readonly Lazy<GptSoVitsPinyin> SharedInstance = new(() => new GptSoVitsPinyin(), isThreadSafe: true);

    /// <summary>One item's reading in the three styles.</summary>
    internal readonly record struct Reading(string Tone3, string Initial, string Final);

    private readonly Dictionary<string, Reading> _characters = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Reading[]> _phrases = new(StringComparer.Ordinal);
    private readonly HashSet<string> _phrasePrefixes = new(StringComparer.Ordinal);

    public static GptSoVitsPinyin Default => SharedInstance.Value;

    private static Reading Parse(string triple)
    {
        var parts = triple.Split(',');
        return new Reading(parts[0], parts[1], parts[2]);
    }

    private GptSoVitsPinyin()
    {
        foreach (var line in ChineseText.ReadLines("gptsovits_pypinyin_chars.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            _characters[line.Substring(0, tab)] = Parse(line.Substring(tab + 1));
        }
        foreach (var line in ChineseText.ReadLines("gptsovits_pypinyin_phrases.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            string phrase = line.Substring(0, tab);
            _phrases[phrase] = line.Substring(tab + 1).Split(' ').Select(Parse).ToArray();
            var prefix = new StringBuilder();
            foreach (var c in ChineseText.Codepoints(phrase))
            {
                prefix.Append(c);
                _phrasePrefixes.Add(prefix.ToString());
            }
        }
    }

    /// <summary>Each item's reading: a Han syllable, or a run of other text as written (in every style).</summary>
    public List<Reading> Read(string text)
    {
        var result = new List<Reading>();
        var runs = new List<(bool Han, List<string> Text)>();
        foreach (var c in ChineseText.Codepoints(text))
        {
            bool han = PypinyinReader.IsHan(c);
            if (runs.Count == 0 || runs[runs.Count - 1].Han != han) runs.Add((han, new List<string>()));
            runs[runs.Count - 1].Text.Add(c);
        }
        foreach (var (han, run) in runs)
        {
            if (!han)
            {
                string kept = string.Concat(run);
                result.Add(new Reading(kept, kept, kept));
                continue;
            }
            foreach (var piece in MaximumMatching(run))
            {
                if (piece.Count > 1 && _phrases.TryGetValue(string.Concat(piece), out var readings)) result.AddRange(readings);
                else result.AddRange(piece.Select(c => _characters.TryGetValue(c, out var r) ? r : new Reading(c, c, c)));
            }
        }
        return result;
    }

    // pypinyin.seg.mmseg.Seg(no_non_phrases=True).cut
    private IEnumerable<List<string>> MaximumMatching(List<string> text)
    {
        int start = 0;
        while (start < text.Count)
        {
            int lastValid = 0;
            bool broke = false;
            for (int index = start; index < text.Count; index++)
            {
                string candidate = string.Concat(text.GetRange(start, index - start + 1));
                if (_phrasePrefixes.Contains(candidate))
                {
                    if (_phrases.ContainsKey(candidate)) lastValid = index - start + 1;
                }
                else
                {
                    if (lastValid > 0)
                    {
                        yield return text.GetRange(start, lastValid);
                        start += lastValid;
                    }
                    else
                    {
                        yield return text.GetRange(start, 1);
                        start += 1;
                    }
                    broke = true;
                    break;
                }
            }
            if (broke) continue;
            if (lastValid > 0)
            {
                yield return text.GetRange(start, lastValid);
                start += lastValid;
            }
            else
            {
                string remain = string.Concat(text.GetRange(start, text.Count - start));
                if (_phrases.ContainsKey(remain)) yield return text.GetRange(start, text.Count - start);
                else
                    foreach (var c in text.GetRange(start, text.Count - start)) yield return new List<string> { c };
                yield break;
            }
        }
    }

    // ------------------------------------------------------------------ pypinyin.contrib.tone_convert

    private static readonly string[] Initials =
        { "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s" };

    private static readonly HashSet<string> Finals = new(StringComparer.Ordinal)
    {
        "i", "u", "ü", "a", "ia", "ua", "o", "uo", "e", "ie", "üe", "ai", "uai", "ei", "uei", "ao", "iao", "ou", "iou",
        "an", "ian", "uan", "üan", "en", "in", "uen", "ün", "ang", "iang", "uang", "eng", "ing", "ueng", "ong", "iong",
        "er", "ê",
    };

    // phonetic_symbol.phonetic_symbol: a toned letter to its letter and tone number (ü is written v).
    private static readonly (string Symbol, string Number)[] PhoneticSymbols =
    {
        ("m̄", "m1"), ("m̀", "m4"), ("ê̄", "ê1"), ("ê̌", "ê3"),
        ("ā", "a1"), ("á", "a2"), ("ǎ", "a3"), ("à", "a4"), ("ē", "e1"), ("é", "e2"), ("ě", "e3"),
        ("è", "e4"), ("ō", "o1"), ("ó", "o2"), ("ǒ", "o3"), ("ò", "o4"), ("ī", "i1"), ("í", "i2"),
        ("ǐ", "i3"), ("ì", "i4"), ("ū", "u1"), ("ú", "u2"), ("ǔ", "u3"), ("ù", "u4"), ("ü", "v"),
        ("ǖ", "v1"), ("ǘ", "v2"), ("ǚ", "v3"), ("ǜ", "v4"), ("ń", "n2"), ("ň", "n3"), ("ǹ", "n4"),
        ("ḿ", "m2"), ("ế", "ê2"), ("ề", "ê4"),
    };

    // replace_symbol_to_number: the single-character symbols, then the two-character ones.
    private static string SymbolsToNumbers(string pinyin)
    {
        var output = new StringBuilder(pinyin.Length);
        foreach (char c in pinyin)
        {
            string s = c.ToString();
            var hit = PhoneticSymbols.FirstOrDefault(p => p.Symbol == s);
            output.Append(hit.Symbol is null ? s : hit.Number);
        }
        string value = output.ToString();
        foreach (var (symbol, number) in PhoneticSymbols.Where(p => p.Symbol.Length != 1)) value = value.Replace(symbol, number);
        return value;
    }

    private static readonly Regex Digit = new(@"\d");

    /// <summary><c>to_initials(pinyin)</c> (strict): the first initial the reading starts with, else empty.</summary>
    public static string ToInitials(string pinyin) => Initials.FirstOrDefault(i => pinyin.StartsWith(i, StringComparison.Ordinal)) ?? "";

    private static string InitialOf(string pinyin, bool strict)
    {
        string initial = ToInitials(pinyin);
        if (initial.Length > 0 || strict) return initial;
        return pinyin.StartsWith("y", StringComparison.Ordinal) ? "y" : pinyin.StartsWith("w", StringComparison.Ordinal) ? "w" : "";
    }

    /// <summary><c>to_finals_tone3(pinyin, neutral_tone_with_five)</c> (strict, ü written v).</summary>
    public static string ToFinalsTone3(string pinyin, bool neutralToneWithFive)
    {
        pinyin = pinyin.Replace("5", "");
        string finals = ToFinals(pinyin);
        if (finals.Length == 0) return finals;
        var number = Digit.Match(SymbolsToNumbers(pinyin));
        if (!number.Success) return neutralToneWithFive ? finals + "5" : finals;
        return finals + number.Value;
    }

    // to_finals (strict, v_to_u=False)
    private static string ToFinals(string pinyin)
    {
        string plain = Digit.Replace(SymbolsToNumbers(pinyin), "").Replace("v", "ü");
        // get_finals(strict=True)
        string converted = ConvertFinals(plain);
        string finals = converted.Substring(InitialOf(converted, strict: true).Length);
        if (!Finals.Contains(finals))
        {
            finals = converted.Substring(InitialOf(converted, strict: false).Length);
            if (!Finals.Contains(finals)) finals = "";
        }
        return finals.Replace("ü", "v");
    }

    private static readonly Regex UvPattern = new("^(j|q|x)(u)(.*)$");
    private static readonly Regex IuPattern = new("^([a-z]+)(iu)$");
    private static readonly Regex UiPattern = new("([a-z]+)(ui)$");
    private static readonly Regex UnPattern = new("([a-z]+)(un)$");

    // pypinyin.standard.convert_finals on a reading without tone marks.
    private static string ConvertFinals(string pinyin)
    {
        // convert_zero_consonant
        string raw = pinyin;
        if (raw.StartsWith("y", StringComparison.Ordinal))
        {
            string noY = pinyin.Substring(1);
            string? first = noY.Length > 0 ? noY.Substring(0, 1) : null;
            if (first == "u") pinyin = "ü" + pinyin.Substring(2);
            else if (first == "i") pinyin = noY;
            else pinyin = "i" + noY;
        }
        if (raw.StartsWith("w", StringComparison.Ordinal))
        {
            string noW = pinyin.Substring(1);
            string? first = noW.Length > 0 ? noW.Substring(0, 1) : null;
            pinyin = first == "u" ? pinyin.Substring(1) : "u" + pinyin.Substring(1);
        }
        if (!Finals.Contains(pinyin)) pinyin = raw;
        pinyin = UvPattern.Replace(pinyin, m => m.Groups[1].Value + "ü" + m.Groups[3].Value);
        pinyin = IuPattern.Replace(pinyin, m => m.Groups[1].Value + "iou");
        pinyin = UiPattern.Replace(pinyin, m => m.Groups[1].Value + "uei");
        return UnPattern.Replace(pinyin, m => m.Groups[1].Value + "uen");
    }
}

/// <summary>
/// GPT-SoVITS's Chinese text normalization (<c>text/zh_normalization</c>, from PaddleSpeech; Apache-2.0):
/// <c>TextNormalizer.normalize</c>, which splits text into sentences and writes out dates, times, ranges, temperatures,
/// units, arithmetic, fractions, percentages, phone numbers and numbers in Chinese.
/// </summary>
internal static class GptSoVitsChineseNormalizer
{
    private static readonly Regex SentenceSplitter = new("([：、，；。？！,;?!][”’]?)");
    private static readonly Regex SplitFilter = new(@"[——《》【】<>{}()（）#&@“”^_|\\]");
    private static readonly Regex PostFilter = new(@"[-——《》【】<=>{}()（）#&@“”^_|\\]");
    private static readonly Regex Newlines = new(@"\n+");

    /// <summary><c>TextNormalizer().normalize(text)</c>: the normalized sentences.</summary>
    public static List<string> Normalize(string text) => Split(text).Select(NormalizeSentence).ToList();

    private static List<string> Split(string text)
    {
        text = text.Replace(" ", "");
        text = SplitFilter.Replace(text, "");
        text = SentenceSplitter.Replace(text, "$1\n");
        text = PythonText.Strip(text);
        return Newlines.Split(text).Select(PythonText.Strip).ToList();
    }

    private static readonly (string From, string To)[] PostReplacements =
    {
        ("/", "每"), ("①", "一"), ("②", "二"), ("③", "三"), ("④", "四"), ("⑤", "五"), ("⑥", "六"), ("⑦", "七"), ("⑧", "八"), ("⑨", "九"),
        ("⑩", "十"), ("α", "阿尔法"), ("β", "贝塔"), ("γ", "伽玛"), ("Γ", "伽玛"), ("δ", "德尔塔"), ("Δ", "德尔塔"), ("ε", "艾普西龙"),
        ("ζ", "捷塔"), ("η", "依塔"), ("θ", "西塔"), ("Θ", "西塔"), ("ι", "艾欧塔"), ("κ", "喀帕"), ("λ", "拉姆达"), ("Λ", "拉姆达"),
        ("μ", "缪"), ("ν", "拗"), ("ξ", "克西"), ("Ξ", "克西"), ("ο", "欧米克伦"), ("π", "派"), ("Π", "派"), ("ρ", "肉"), ("ς", "西格玛"),
        ("Σ", "西格玛"), ("σ", "西格玛"), ("τ", "套"), ("υ", "宇普西龙"), ("φ", "服艾"), ("Φ", "服艾"), ("χ", "器"), ("ψ", "普赛"),
        ("Ψ", "普赛"), ("ω", "欧米伽"), ("Ω", "欧米伽"), ("+", "加"), ("-", "减"), ("×", "乘"), ("÷", "除"), ("=", "等"),
    };

    private static string PostReplace(string sentence)
    {
        foreach (var (from, to) in PostReplacements) sentence = sentence.Replace(from, to);
        return PostFilter.Replace(sentence, "");
    }

    // F2H_ASCII_LETTERS, F2H_DIGITS, F2H_SPACE
    private static string FullWidthToHalfWidth(string sentence)
    {
        var output = new StringBuilder(sentence.Length);
        foreach (char c in sentence)
        {
            char half = (char)(c - 65248);
            if (c >= 0xFF01 && ((half >= 'a' && half <= 'z') || (half >= 'A' && half <= 'Z') || (half >= '0' && half <= '9'))) output.Append(half);
            else output.Append(c == '　' ? ' ' : c);
        }
        return output.ToString();
    }

    /// <summary><c>TextNormalizer.normalize_sentence</c>.</summary>
    public static string NormalizeSentence(string sentence)
    {
        sentence = GptSoVitsChineseData.Instance.ToSimplified(sentence);
        sentence = FullWidthToHalfWidth(sentence);
        sentence = Date.Replace(sentence, ReplaceDate);
        sentence = Date2.Replace(sentence, ReplaceDate2);
        sentence = TimeRange.Replace(sentence, m => ReplaceTime(m, range: true));
        sentence = Time.Replace(sentence, m => ReplaceTime(m, range: false));
        sentence = ToRange.Replace(sentence, m => m.Value.Replace("~", "至"));
        sentence = Temperature.Replace(sentence, ReplaceTemperature);
        sentence = ReplaceMeasure(sentence);
        while (Asmd.IsMatch(sentence)) sentence = Asmd.Replace(sentence, ReplaceAsmd);
        sentence = Power.Replace(sentence, ReplacePower);
        sentence = Fraction.Replace(sentence, ReplaceFraction);
        sentence = Percentage.Replace(sentence, m => (m.Groups[1].Value.Length > 0 ? "负" : "") + "百分之" + NumberToString(m.Groups[2].Value));
        sentence = MobilePhone.Replace(sentence, m => PhoneToString(m.Value, mobile: true));
        sentence = Telephone.Replace(sentence, m => PhoneToString(m.Value, mobile: false));
        sentence = NationalUniformNumber.Replace(sentence, m => PhoneToString(m.Value, mobile: false));
        sentence = Range.Replace(sentence, m => Number.Replace(m.Groups[1].Value, ReplaceNumber) + "到" + Number.Replace(m.Groups[6].Value, ReplaceNumber));
        sentence = Integer.Replace(sentence, m => "负" + NumberToString(m.Groups[2].Value));
        sentence = VersionNumber.Replace(sentence, ReplaceVersion);
        sentence = DecimalNumber.Replace(sentence, ReplaceNumber);
        sentence = PositiveQuantifiers.Replace(sentence, ReplacePositiveQuantifier);
        sentence = DefaultNumber.Replace(sentence, m => VerbalizeDigit(m.Value, alternateOne: true));
        sentence = Number.Replace(sentence, ReplaceNumber);
        return PostReplace(sentence);
    }

    // ------------------------------------------------------------------ num.py

    private const string DigitNames = "零一二三四五六七八九";
    private static readonly (int Power, string Name)[] Units = { (1, "十"), (2, "百"), (3, "千"), (4, "万"), (8, "亿") };

    private static string Digit(char c)
    {
        if (c < '0' || c > '9') throw new KeyNotFoundException($"'{c}' is not one of DIGITS' keys.");
        return DigitNames[c - '0'].ToString();
    }

    private static List<string> Value(string valueString, bool useZero = true)
    {
        string stripped = valueString.TrimStart('0');
        if (stripped.Length == 0) return new List<string>();
        if (stripped.Length == 1)
        {
            if (useZero && stripped.Length < valueString.Length) return new List<string> { "零", Digit(stripped[0]) };
            return new List<string> { Digit(stripped[0]) };
        }
        var (power, name) = Enumerable.Reverse(Units).First(u => u.Power < stripped.Length);
        var result = Value(valueString.Substring(0, valueString.Length - power));
        result.Add(name);
        result.AddRange(Value(valueString.Substring(valueString.Length - power)));
        return result;
    }

    internal static string VerbalizeCardinal(string valueString)
    {
        if (valueString.Length == 0) return "";
        valueString = valueString.TrimStart('0');
        if (valueString.Length == 0) return "零";
        var symbols = Value(valueString);
        if (symbols.Count >= 2 && symbols[0] == "一" && symbols[1] == "十") symbols.RemoveAt(0);
        return string.Concat(symbols);
    }

    internal static string VerbalizeDigit(string valueString, bool alternateOne = false)
    {
        string result = string.Concat(valueString.Select(Digit));
        return alternateOne ? result.Replace("一", "幺") : result;
    }

    internal static string NumberToString(string valueString)
    {
        var parts = valueString.Split('.');
        if (parts.Length > 2) throw new ArgumentException($"The value string: '${valueString}' has more than one point in it.");
        string integer = parts[0], fraction = parts.Length == 2 ? parts[1] : "";
        string result = VerbalizeCardinal(integer);
        fraction = fraction.EndsWith("0", StringComparison.Ordinal) ? fraction.TrimEnd('0') + "0" : fraction.TrimEnd('0');
        if (fraction.Length > 0)
        {
            if (result.Length == 0) result = "零";
            result += "点" + VerbalizeDigit(fraction);
        }
        return result;
    }

    private const string ComQuantifiers =
        "(处|台|架|枚|趟|幅|平|方|堵|间|床|株|批|项|例|列|篇|栋|注|亩|封|艘|把|目|套|段|人|所|朵|匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|毫|厘|(公)分|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|小时|旬|纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块|元|(亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|美|)元|(亿|千万|百万|万|千|百|十|)吨|(亿|千万|百万|万|千|百|)块|角|毛|分)";

    private const string Superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ";
    private static readonly Regex Fraction = new(@"(-?)(\d+)/(\d+)");
    private static readonly Regex Percentage = new(@"(-?)(\d+(\.\d+)?)%");
    private static readonly Regex Integer = new(@"(-)(\d+)");
    private static readonly Regex DefaultNumber = new(@"\d{3}\d*");
    private static readonly Regex Asmd = new(
        @"((-?)((\d+)(\.\d+)?[" + Superscripts + @"]*)|(\.\d+[" + Superscripts + @"]*)|([A-Za-z][" + Superscripts + @"]*))([+\-×÷=])((-?)((\d+)(\.\d+)?[" +
        Superscripts + @"]*)|(\.\d+[" + Superscripts + @"]*)|([A-Za-z][" + Superscripts + @"]*))");
    private static readonly Regex Power = new("[" + Superscripts + "]+");
    private static readonly Regex DecimalNumber = new(@"(-?)((\d+)(\.\d+))|(\.(\d+))");
    private static readonly Regex PositiveQuantifiers = new(@"(\d+)([多余几+])?" + ComQuantifiers);
    private static readonly Regex Number = new(@"(-?)((\d+)(\.\d+)?)|(\.(\d+))");
    private static readonly Regex Range = new(@"(?<![\d+\-×÷=])((-?)((\d+)(\.\d+)?))[-~]((-?)((\d+)(\.\d+)?))(?![\d+\-×÷=])");
    private const string UnitAlternatives = "(%|°C|℃|度|摄氏度|cm2|cm²|cm3|cm³|cm|db|ds|kg|km|m2|m²|m³|m3|ml|m|mm|s)";
    private static readonly Regex ToRange = new(@"((-?)((\d+)(\.\d+)?)|(\.(\d+)))" + UnitAlternatives + @"[~]((-?)((\d+)(\.\d+)?)|(\.(\d+)))" + UnitAlternatives);
    private static readonly Regex VersionNumber = new(@"((\d+)(\.\d+)(\.\d+)?(\.\d+)+)");

    private static string ReplaceFraction(Match m)
        => (m.Groups[1].Value.Length > 0 ? "负" : "") + NumberToString(m.Groups[3].Value) + "分之" + NumberToString(m.Groups[2].Value);

    private static string ReplaceAsmd(Match m)
    {
        string op = m.Groups[8].Value switch { "+" => "加", "-" => "减", "×" => "乘", "÷" => "除", _ => "等于" };
        return m.Groups[1].Value + op + m.Groups[9].Value;
    }

    private static string ReplacePower(Match m)
    {
        var power = new StringBuilder();
        foreach (char c in m.Value) power.Append(c switch { 'ˣ' => "x", 'ʸ' => "y", 'ⁿ' => "n", _ => ((char)('0' + "⁰¹²³⁴⁵⁶⁷⁸⁹".IndexOf(c))).ToString() });
        return "的" + power + "次方";
    }

    private static string ReplaceNumber(Match m)
    {
        if (m.Groups[5].Value.Length > 0) return NumberToString(m.Groups[5].Value);
        return (m.Groups[1].Value.Length > 0 ? "负" : "") + NumberToString(m.Groups[2].Value);
    }

    private static string ReplacePositiveQuantifier(Match m)
    {
        string more = m.Groups[2].Value == "+" ? "多" : m.Groups[2].Value;
        string number = NumberToString(m.Groups[1].Value);
        if (number == "二") number = "两";
        return number + more + m.Groups[3].Value;
    }

    private static string ReplaceVersion(Match m)
    {
        var result = new StringBuilder();
        foreach (char c in m.Groups[1].Value) result.Append(c == '.' ? "点" : NumberToString(c.ToString()));
        return result.ToString();
    }

    // ------------------------------------------------------------------ chronology.py

    private static readonly Regex Time = new(@"([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?");
    private static readonly Regex TimeRange = new(@"([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?(~|-)([0-1]?[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?");
    private static readonly Regex Date = new(@"(\d{4}|\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?");
    private static readonly Regex Date2 = new(@"(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])");

    private static string TimeNumberToString(string number)
    {
        string result = NumberToString(number.TrimStart('0'));
        return number.StartsWith("0", StringComparison.Ordinal) ? "零" + result : result;
    }

    private static string ReplaceTime(Match m, bool range)
    {
        string hour = m.Groups[1].Value, minute = m.Groups[2].Value, second = m.Groups[4].Value;
        string result = NumberToString(hour) + "点";
        if (minute.TrimStart('0').Length > 0)
            result += int.Parse(minute, CultureInfo.InvariantCulture) == 30 ? "半" : TimeNumberToString(minute) + "分";
        if (second.Length > 0 && second.TrimStart('0').Length > 0) result += TimeNumberToString(second) + "秒";
        if (range)
        {
            string hour2 = m.Groups[6].Value, minute2 = m.Groups[7].Value, second2 = m.Groups[9].Value;
            result += "至" + NumberToString(hour2) + "点";
            // As the reference: the second minute's half hour is decided by the first minute.
            if (minute2.TrimStart('0').Length > 0)
                result += int.Parse(minute, CultureInfo.InvariantCulture) == 30 ? "半" : TimeNumberToString(minute2) + "分";
            if (second2.Length > 0 && second2.TrimStart('0').Length > 0) result += TimeNumberToString(second2) + "秒";
        }
        return result;
    }

    private static string ReplaceDate(Match m)
    {
        string year = m.Groups[1].Value, month = m.Groups[3].Value, day = m.Groups[5].Value;
        string result = "";
        if (year.Length > 0) result += VerbalizeDigit(year) + "年";
        if (month.Length > 0) result += VerbalizeCardinal(month) + "月";
        if (day.Length > 0) result += VerbalizeCardinal(day) + m.Groups[9].Value;
        return result;
    }

    private static string ReplaceDate2(Match m)
    {
        string year = m.Groups[1].Value, month = m.Groups[3].Value, day = m.Groups[4].Value;
        string result = "";
        if (year.Length > 0) result += VerbalizeDigit(year) + "年";
        if (month.Length > 0) result += VerbalizeCardinal(month) + "月";
        if (day.Length > 0) result += VerbalizeCardinal(day) + "日";
        return result;
    }

    // ------------------------------------------------------------------ quantifier.py and phonecode.py

    private static readonly Regex Temperature = new(@"(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)");

    // As the reference: the unit is read from group 3 (the decimal part), so it is always "度".
    private static string ReplaceTemperature(Match m)
        => (m.Groups[1].Value.Length > 0 ? "零下" : "") + NumberToString(m.Groups[2].Value) + (m.Groups[3].Value == "摄氏度" ? "摄氏度" : "度");

    private static readonly (string Notation, string Name)[] Measures =
    {
        ("cm2", "平方厘米"), ("cm²", "平方厘米"), ("cm3", "立方厘米"), ("cm³", "立方厘米"), ("cm", "厘米"), ("db", "分贝"), ("ds", "毫秒"),
        ("kg", "千克"), ("km", "千米"), ("m2", "平方米"), ("m²", "平方米"), ("m³", "立方米"), ("m3", "立方米"), ("ml", "毫升"), ("m", "米"),
        ("mm", "毫米"), ("s", "秒"),
    };

    private static string ReplaceMeasure(string sentence)
    {
        foreach (var (notation, name) in Measures)
            if (sentence.Contains(notation)) sentence = sentence.Replace(notation, name);
        return sentence;
    }

    private static readonly Regex MobilePhone = new(@"(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)");
    private static readonly Regex Telephone = new(@"(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)");
    private static readonly Regex NationalUniformNumber = new(@"(400)(-)?\d{3}(-)?\d{4}");

    private static string PhoneToString(string phone, bool mobile)
    {
        var parts = mobile
            ? phone.Trim('+').Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            : phone.Split('-');
        return string.Join("，", parts.Select(p => VerbalizeDigit(p, alternateOne: true)));
    }
}
