using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// g2pkk 0.1.2 (kyubyong/g2pK as packaged for MeloTTS; Apache-2.0) with its defaults (prescriptive, vowels kept,
/// syllables recomposed): idioms, English words through the CMU dictionary, mecab-ko's part-of-speech annotations,
/// numbers spelled out, then the jamo-level special rules, the coda-onset table and the linking rules.
/// </summary>
/// <remarks>
/// Where g2pkk iterates a Python set of matches and replaces them one after another (English words, numbers before
/// bound nouns), the order decides the result only when one match contains another; that order varies with Python's
/// hash seed, and here it is the order of first appearance.
/// </remarks>
internal sealed class G2pKk
{
    private static readonly Lazy<G2pKk> SharedInstance = new(() => new G2pKk("g2pkk012_table.csv.gz", linkAcrossSpaces: true), isThreadSafe: true);
    private static readonly Lazy<G2pKk> G2pk2Instance = new(() => new G2pKk("g2pk2003_table.csv.gz", linkAcrossSpaces: false), isThreadSafe: true);
    private readonly List<(string From, string To)> _idioms = new();
    private readonly List<(Regex Pattern, string Replacement)> _table = new();
    private readonly bool _linkAcrossSpaces;

    public static G2pKk Default => SharedInstance.Value;

    /// <summary>
    /// g2pk2 0.0.3 (GPT-SoVITS's Korean G2P): g2pkk with two cells of its coda-onset table corrected (ᆰ before ᄏ, ᆻ
    /// before ᄑ) and the linking of a coda to an ᄋ onset across a space (link3) commented out.
    /// </summary>
    public static G2pKk G2pk2 => G2pk2Instance.Value;

    private G2pKk(string table, bool linkAcrossSpaces)
    {
        _linkAcrossSpaces = linkAcrossSpaces;
        foreach (var raw in ChineseText.ReadLines("g2pkk012_idioms.txt.gz"))
        {
            string line = PythonText.Strip(raw.Split('#')[0]);
            if (!line.Contains("===")) continue;
            var parts = line.Split(new[] { "===" }, StringSplitOptions.None);
            if (parts.Length != 2) throw new InvalidOperationException($"idioms.txt line '{line}' does not split in two.");
            _idioms.Add((parts[0], parts[1]));
        }
        var lines = ChineseText.ReadLines(table).ToList();
        var onsets = lines[0].Split(',');
        foreach (var line in lines.Skip(1))
        {
            var cols = line.Split(',');
            string coda = cols[0];
            for (int i = 1; i < onsets.Length; i++)
            {
                string cell = cols[i];
                if (cell.Length == 0) continue;
                string replacement = cell.Contains("(") ? cell.Split('(')[0] : cell;
                _table.Add((new Regex(PythonPattern(coda + onsets[i]), RegexOptions.CultureInvariant), PythonReplacement(replacement)));
            }
        }
    }

    // The table's patterns use \W (Python's: not alphanumeric or underscore) and its templates \1.
    private static string PythonPattern(string pattern) => pattern.Replace(@"\W", "[^" + PythonText.WordClass + "]");
    private static string PythonReplacement(string template) => template.Replace("$", "$$").Replace(@"\1", "${1}");

    /// <summary><c>G2p()(text)</c>.</summary>
    public string Convert(string text)
    {
        foreach (var (from, to) in _idioms) text = text.Replace(from, to);
        text = ConvertEnglish(text);
        text = Annotate(text);
        text = ConvertNumbers(text);
        string inp = HangulToJamo(text);
        inp = Special(inp);
        inp = Regex.Replace(inp, "/[PJEB]", "");
        foreach (var (pattern, replacement) in _table) inp = pattern.Replace(inp, replacement);
        inp = Link(inp);
        return Compose(inp);
    }

    // ------------------------------------------------------------------ jamo (jamo 0.4.1)

    /// <summary><c>h2j</c>: each Hangul syllable as its conjoining lead, vowel and (if any) tail.</summary>
    internal static string HangulToJamo(string text)
    {
        var sb = new StringBuilder(text.Length * 3);
        foreach (char c in text)
        {
            if (c >= 0xAC00 && c <= 0xD7A3)
            {
                int s = c - 0xAC00;
                sb.Append((char)(0x1100 + s / (21 * 28)));
                sb.Append((char)(0x1161 + s % (21 * 28) / 28));
                if (s % 28 != 0) sb.Append((char)(0x11A7 + s % 28));
            }
            else sb.Append(c);
        }
        return sb.ToString();
    }

    private static string JamoToHangul(char lead, char vowel, char? tail = null) =>
        ((char)(0xAC00 + ((lead - 0x1100) * 21 + (vowel - 0x1161)) * 28 + (tail is null ? 0 : tail.Value - 0x11A7))).ToString();

    private static readonly Regex LoneVowel = new("(^|[^ᄀ-ᄒ])([ᅡ-ᅵ])", RegexOptions.CultureInvariant);
    private static readonly Regex FullSyllable = new("[ᄀ-ᄒ][ᅡ-ᅵ][ᆨ-ᇂ]", RegexOptions.CultureInvariant);
    private static readonly Regex OpenSyllable = new("[ᄀ-ᄒ][ᅡ-ᅵ]", RegexOptions.CultureInvariant);

    /// <summary><c>compose</c>: a vowel without a lead given ᄋ, then lead-vowel(-tail) runs made syllables.</summary>
    internal static string Compose(string letters)
    {
        letters = LoneVowel.Replace(letters, "${1}ᄋ${2}");
        letters = FullSyllable.Replace(letters, m => JamoToHangul(m.Value[0], m.Value[1], m.Value[2]));
        return OpenSyllable.Replace(letters, m => JamoToHangul(m.Value[0], m.Value[1]));
    }

    // ------------------------------------------------------------------ English (english.py, utils.py)

    private static readonly Regex EnglishWord = new("[A-Za-z']+", RegexOptions.CultureInvariant);

    private static List<string> Adjust(string[] arpabets)
    {
        string s = " " + string.Join(" ", arpabets) + " $";
        s = Regex.Replace(s, @"\d", "");
        s = s.Replace(" T S ", " TS ").Replace(" D Z ", " DZ ").Replace(" AW ER ", " AWER ")
            .Replace(" IH R $", " IH ER ").Replace(" EH R $", " EH ER ").Replace(" $", "");
        return s.Trim('$', ' ').Split(new[] { ' ', '\t', '\n', '\r', '\f', '\v' }, StringSplitOptions.RemoveEmptyEntries).ToList();
    }

    private static readonly Dictionary<string, string> Choseong = new(StringComparer.Ordinal)
    {
        ["B"] = "ᄇ", ["CH"] = "ᄎ", ["D"] = "ᄃ", ["DH"] = "ᄃ", ["DZ"] = "ᄌ", ["F"] = "ᄑ", ["G"] = "ᄀ", ["HH"] = "ᄒ",
        ["JH"] = "ᄌ", ["K"] = "ᄏ", ["L"] = "ᄅ", ["M"] = "ᄆ", ["N"] = "ᄂ", ["NG"] = "ᄋ", ["P"] = "ᄑ", ["R"] = "ᄅ",
        ["S"] = "ᄉ", ["SH"] = "ᄉ", ["T"] = "ᄐ", ["TH"] = "ᄉ", ["TS"] = "ᄎ", ["V"] = "ᄇ", ["W"] = "W", ["Y"] = "Y",
        ["Z"] = "ᄌ", ["ZH"] = "ᄌ",
    };
    private static readonly Dictionary<string, string> Jungseong = new(StringComparer.Ordinal)
    {
        ["AA"] = "ᅡ", ["AE"] = "ᅢ", ["AH"] = "ᅥ", ["AO"] = "ᅩ", ["AW"] = J("ᅡ우"), ["AWER"] = J("ᅡ워"), ["AY"] = J("ᅡ이"),
        ["EH"] = "ᅦ", ["ER"] = "ᅥ", ["EY"] = J("ᅦ이"), ["IH"] = "ᅵ", ["IY"] = "ᅵ", ["OW"] = "ᅩ", ["OY"] = J("ᅩ이"),
        ["UH"] = "ᅮ", ["UW"] = "ᅮ",
    };
    private static readonly Dictionary<string, string> Jongseong = new(StringComparer.Ordinal)
    {
        ["B"] = "ᆸ", ["CH"] = "ᆾ", ["D"] = "ᆮ", ["DH"] = "ᆮ", ["F"] = "ᇁ", ["G"] = "ᆨ", ["HH"] = "ᇂ", ["JH"] = "ᆽ",
        ["K"] = "ᆨ", ["L"] = "ᆯ", ["M"] = "ᆷ", ["N"] = "ᆫ", ["NG"] = "ᆼ", ["P"] = "ᆸ", ["R"] = "ᆯ", ["S"] = "ᆺ",
        ["SH"] = "ᆺ", ["T"] = "ᆺ", ["TH"] = "ᆺ", ["V"] = "ᆸ", ["W"] = "ᆼ", ["Y"] = "ᆼ", ["Z"] = "ᆽ", ["ZH"] = "ᆽ",
    };

    private static string Map(Dictionary<string, string> map, string p) => map.TryGetValue(p, out var v) ? v : p;

    private static readonly (string, string)[] Reconstructions =
    {
        ("그W", "ᄀW"), ("흐W", "ᄒW"), ("크W", "ᄏW"), ("ᄂYᅥ", "니어"), ("ᄃYᅥ", "디어"), ("ᄅYᅥ", "리어"), ("Yᅵ", "ᅵ"),
        ("Yᅡ", "ᅣ"), ("Yᅢ", "ᅤ"), ("Yᅥ", "ᅧ"), ("Yᅦ", "ᅨ"), ("Yᅩ", "ᅭ"), ("Yᅮ", "ᅲ"), ("Wᅡ", "ᅪ"), ("Wᅢ", "ᅫ"),
        ("Wᅥ", "ᅯ"), ("Wᅩ", "ᅯ"), ("Wᅮ", "ᅮ"), ("Wᅦ", "ᅰ"), ("Wᅵ", "ᅱ"), ("ᅳᅵ", "ᅴ"), ("Y", "ᅵ"), ("W", "ᅮ"),
    };

    private static readonly string[] ShortVowels = { "AE", "AH", "AX", "EH", "IH", "IX", "UH" };

    // Python's `a in b` for strings (a substring test), and slicing that never throws.
    private static bool In(string a, string b) => b.Contains(a);
    private static string First(string s) => s.Substring(0, 1);
    private static string FirstTwo(string s) => s.Length >= 2 ? s.Substring(0, 2) : s;

    private static string ConvertEnglish(string text)
    {
        var words = new List<string>();
        foreach (Match m in EnglishWord.Matches(text))
            if (!words.Contains(m.Value)) words.Add(m.Value);
        foreach (var englishWord in words)
        {
            if (!G2pEn.Default.TryGetCmu(englishWord.ToLowerInvariant(), out var arpabets)) continue;
            var phonemes = Adjust(arpabets);
            var ret = new StringBuilder();
            const string vowels = "AEIOUY", consonants = "BCDFGHJKLMNPQRSTVWXZ", finalOrConsonants = "$BCDFGHJKLMNPQRSTVWXZ";
            for (int i = 0; i < phonemes.Count; i++)
            {
                string p = phonemes[i];
                string prev = i > 0 ? phonemes[i - 1] : "^";
                string next = i < phonemes.Count - 1 ? phonemes[i + 1] : "$";
                string next2 = i < phonemes.Count - 2 ? phonemes[i + 1] : "$";   // g2pkk's own index
                if (In(p, "PTK"))
                {
                    if (ShortVowels.Contains(FirstTwo(prev)) && next == "$") ret.Append(Map(Jongseong, p));
                    else if (ShortVowels.Contains(FirstTwo(prev)) && !In(First(next), "AEIOULRMN")) ret.Append(Map(Jongseong, p));
                    else if (In(First(next), "$BCDFGHJKLMNPQRSTVWXYZ")) ret.Append(Map(Choseong, p)).Append("ᅳ");
                    else ret.Append(Map(Choseong, p));
                }
                else if (In(p, "BDG"))
                {
                    ret.Append(Map(Choseong, p));
                    if (In(First(next), finalOrConsonants)) ret.Append("ᅳ");
                }
                else if (p is "S" or "Z" or "F" or "V" or "TH" or "DH" or "SH" or "ZH")
                {
                    ret.Append(Map(Choseong, p));
                    if (p is "S" or "Z" or "F" or "V" or "TH" or "DH")
                    {
                        if (In(First(next), finalOrConsonants)) ret.Append("ᅳ");
                    }
                    else if (p == "SH")
                    {
                        if (In(First(next), "$")) ret.Append("ᅵ");
                        else if (In(First(next), consonants)) ret.Append("ᅲ");
                        else ret.Append("Y");
                    }
                    else if (In(First(next), finalOrConsonants)) ret.Append("ᅵ");
                }
                else if (p is "TS" or "DZ" or "CH" or "JH")
                {
                    ret.Append(Map(Choseong, p));
                    if (In(First(next), finalOrConsonants)) ret.Append(p is "TS" or "DZ" ? "ᅳ" : "ᅵ");
                }
                else if (p is "M" or "N" or "NG")
                {
                    ret.Append(In(p, "MN") && In(First(next), vowels) ? Map(Choseong, p) : Map(Jongseong, p));
                }
                else if (p == "L")
                {
                    if (prev == "^") ret.Append(Map(Choseong, p));
                    else if (In(First(next), "$BCDFGHJKLPQRSTVWXZ")) ret.Append(Map(Jongseong, p));
                    else if (In(prev, "MN")) ret.Append(Map(Choseong, p));
                    else if (In(First(next), vowels)) ret.Append("ᆯᄅ");
                    else if (In(next, "MN") && !In(First(next2), vowels)) ret.Append(J("ᆯ르"));
                }
                else if (p == "ER")
                {
                    if (In(First(prev), vowels)) ret.Append("ᄋ");
                    ret.Append(Map(Jungseong, p));
                    if (In(First(next), vowels)) ret.Append("ᄅ");
                }
                else if (p == "R")
                {
                    if (In(First(next), vowels)) ret.Append(Map(Choseong, p));
                }
                else if (In(First(p), "AEIOU")) ret.Append(Map(Jungseong, p));
                else ret.Append(Map(Choseong, p));
            }
            string word = ret.ToString();
            foreach (var (from, to) in Reconstructions) word = word.Replace(J(from), J(to));
            word = Regex.Replace(Compose(word), "[ᄀ-ᇿ]", "");
            text = text.Replace(englishWord, word);
        }
        return text;
    }

    // ------------------------------------------------------------------ annotate (utils.py)

    private string Annotate(string text)
    {
        var tokens = MeCab.KoDic.Parse(text);
        if (text.Replace(" ", "") != string.Concat(tokens.Select(t => t.Surface))) return text;
        var tags = new StringBuilder();
        foreach (var (surface, tag) in tokens)
        {
            string last = tag.Split('+').Last();
            string mark = last == "NNBC" ? "B" : last.Substring(0, 1);
            tags.Append('_', ChineseText.Codepoints(surface).Count - 1).Append(mark);
        }
        var chars = ChineseText.Codepoints(text);
        var tagChars = ChineseText.Codepoints(tags.ToString());
        for (int i = 0; i < chars.Count; i++)
            if (chars[i] == " ") tagChars.Insert(Math.Min(i, tagChars.Count), " ");
        var annotated = new StringBuilder();
        for (int i = 0; i < Math.Min(chars.Count, tagChars.Count); i++)
        {
            string c = chars[i], t = tagChars[i];
            annotated.Append(c);
            string lastJamo = HangulToJamo(c);
            lastJamo = lastJamo.Substring(lastJamo.Length - 1);
            if (c == "의" && t == "J") annotated.Append("/J");
            else if (t == "E") { if (In(lastJamo, "ᆯ")) annotated.Append("/E"); }
            else if (t == "V") { if (In(lastJamo, "ᆫᆬᆷᆱᆰᆲᆴ")) annotated.Append("/P"); }
            else if (t == "B") annotated.Append("/B");
        }
        return annotated.ToString();
    }

    // ------------------------------------------------------------------ numerals.py

    private const string BoundNouns = "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통";

    /// <summary><c>process_num(num, sino)</c>.</summary>
    internal static string ProcessNumber(string num, bool sino)
    {
        num = num.Replace(",", "");
        if (num == "0") return "영";
        if (!sino && num == "20") return "스무";
        const string digits = "123456789";
        string[] names = { "일", "이", "삼", "사", "오", "육", "칠", "팔", "구" };
        string[] modifiers = { "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉" };
        string[] decimals = { "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔" };
        string Name(char d) => digits.IndexOf(d) is int k && k >= 0 ? names[k] : "";
        var spelled = new List<string>();
        var chars = ChineseText.Codepoints(num);
        string? name = null;   // Python keeps the previous iteration's value where no branch assigns it
        for (int position = 0; position < chars.Count; position++)
        {
            int i = chars.Count - position - 1;
            char digit = chars[position].Length == 1 ? chars[position][0] : '\0';
            int k = digits.IndexOf(digit);
            if (sino)
            {
                if (i == 0) name = Name(digit);
                else if (i == 1) name = (Name(digit) + "십").Replace("일십", "십");
            }
            else
            {
                if (i == 0) name = k >= 0 ? modifiers[k] : "";
                else if (i == 1) name = k >= 0 ? decimals[k] : "";
            }
            if (chars[position] == "0")
            {
                if (i % 4 == 0)
                {
                    var lastThree = spelled.Skip(Math.Max(0, spelled.Count - Math.Min(3, spelled.Count))).ToList();
                    if (spelled.Count == 0 || string.Concat(lastThree).Length == 0)
                    {
                        spelled.Add("");
                        continue;
                    }
                }
                else
                {
                    spelled.Add("");
                    continue;
                }
            }
            string? suffix = i switch
            {
                2 => "백", 3 => "천", 4 => "만", 5 => "십", 6 => "백", 7 => "천", 8 => "억", 9 => "십", 10 => "백", 11 => "천",
                12 => "조", 13 => "십", 14 => "백", 15 => "천", _ => null,
            };
            if (suffix is not null)
            {
                name = Name(digit) + suffix;
                if (i is 2 or 3 or 4 or 5 or 6 or 7) name = name.Replace("일" + suffix, suffix);
            }
            spelled.Add(name ?? throw new InvalidOperationException("process_num: local variable 'name' referenced before assignment"));
        }
        return string.Concat(spelled);
    }

    private static readonly Regex NumberBeforeBoundNoun = new(@"([\d][\d,]*)([ㄱ-힣]+)/B", RegexOptions.CultureInvariant);

    /// <summary><c>convert_num</c>.</summary>
    internal static string ConvertNumbers(string text)
    {
        var tokens = new List<(string Num, string Noun)>();
        foreach (Match m in NumberBeforeBoundNoun.Matches(text))
            if (!tokens.Contains((m.Groups[1].Value, m.Groups[2].Value))) tokens.Add((m.Groups[1].Value, m.Groups[2].Value));
        foreach (var (num, noun) in tokens)
            text = text.Replace($"{num}{noun}/B", $"{ProcessNumber(num, sino: !BoundNouns.Contains(noun))}{noun}/B");
        const string names = "영일이삼사오육칠팔구";
        for (int d = 0; d < 10; d++) text = text.Replace((char)('0' + d), names[d]);
        return text;
    }

    // ------------------------------------------------------------------ special.py (prescriptive), regular.py

    // special.py writes every Korean string as jamo (its "그" is ᄀ+ᅳ, so "([그])" is the class [그]); J() spells the
    // rules the same way.
    private static string J(string s) => HangulToJamo(s);

    private static readonly (Regex Pattern, string Replacement)[] SpecialRules = BuildSpecialRules();

    private static (Regex, string)[] BuildSpecialRules()
    {
        var rules = new List<(string, string)>
        {
            ("([ᄌᄍᄎ])ᅧ", "${1}ᅥ"),                                                  // jyeo
            ("([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ", "${1}ᅵ"),                               // consonant_ui
            ("/J", ""),                                                               // josa_ui (prescriptive)
            ("([그])ᆮᄋ", "${1}ᄉ"), ("([으])[ᆽᆾᇀᇂ]ᄋ", "${1}ᄉ"), ("([으])[ᆿ]ᄋ", "${1}ᄀ"), ("([으])[ᇁ]ᄋ", "${1}ᄇ"),   // jamo
            ("ᆰ/P([ᄀᄁ])", "ᆯᄁ"),                                                     // rieulgiyeok
            ("([ᆲᆴ])/Pᄀ", "${1}ᄁ"), ("([ᆲᆴ])/Pᄃ", "${1}ᄄ"), ("([ᆲᆴ])/Pᄉ", "${1}ᄊ"), ("([ᆲᆴ])/Pᄌ", "${1}ᄍ"),   // rieulbieub
            ("([ᆫᆷ])/Pᄀ", "${1}ᄁ"), ("([ᆫᆷ])/Pᄃ", "${1}ᄄ"), ("([ᆫᆷ])/Pᄉ", "${1}ᄊ"), ("([ᆫᆷ])/Pᄌ", "${1}ᄍ"),   // verb_nieun
            ("ᆬ/Pᄀ", "ᆫᄁ"), ("ᆬ/Pᄃ", "ᆫᄄ"), ("ᆬ/Pᄉ", "ᆫᄊ"), ("ᆬ/Pᄌ", "ᆫᄍ"),
            ("ᆱ/Pᄀ", "ᆷᄁ"), ("ᆱ/Pᄃ", "ᆷᄄ"), ("ᆱ/Pᄉ", "ᆷᄊ"), ("ᆱ/Pᄌ", "ᆷᄍ"),
            ("(바)ᆲ(($|[^ᄋᄒ]))", "${1}ᆸ${2}"), ("(너)ᆲ([ᄌᄍ]ᅮ|[ᄃᄄ]ᅮ)", "${1}ᆸ${2}"),            // balb
            ("ᆮᄋ([ᅵᅧ])", "ᄌ${1}"), ("ᇀᄋ([ᅵᅧ])", "ᄎ${1}"), ("ᆴᄋ([ᅵᅧ])", "ᆯᄎ${1}"), ("ᆮᄒ([ᅵ])", "ᄎ${1}"),   // palatalize
            ("ᆯ/E ᄀ", "ᆯ ᄁ"), ("ᆯ/E ᄃ", "ᆯ ᄄ"), ("ᆯ/E ᄇ", "ᆯ ᄈ"), ("ᆯ/E ᄉ", "ᆯ ᄊ"), ("ᆯ/E ᄌ", "ᆯ ᄍ"),    // modifying_rieul
            ("ᆯ걸", "ᆯ껄"), ("ᆯ밖에", "ᆯ빠께"), ("ᆯ세라", "ᆯ쎄라"), ("ᆯ수록", "ᆯ쑤록"), ("ᆯ지라도", "ᆯ찌라도"),
            ("ᆯ지언정", "ᆯ찌언정"), ("ᆯ진대", "ᆯ찐대"),
        };
        return rules.Select(r => (new Regex(J(r.Item1), RegexOptions.CultureInvariant), J(r.Item2))).ToArray();
    }

    private static string Special(string inp)
    {
        foreach (var (pattern, replacement) in SpecialRules) inp = pattern.Replace(inp, replacement);
        return inp;
    }

    private string Link(string inp)
    {
        string[] codas = { "ᆨ", "ᆩ", "ᆫ", "ᆮ", "ᆯ", "ᆷ", "ᆸ", "ᆺ", "ᆻ", "ᆽ", "ᆾ", "ᆿ", "ᇀ", "ᇁ" };
        string[] onsets = { "ᄀ", "ᄁ", "ᄂ", "ᄃ", "ᄅ", "ᄆ", "ᄇ", "ᄉ", "ᄊ", "ᄌ", "ᄎ", "ᄏ", "ᄐ", "ᄑ" };
        (string, string)[] link2 =
            { ("ᆪᄋ", "ᆨᄊ"), ("ᆬᄋ", "ᆫᄌ"), ("ᆰᄋ", "ᆯᄀ"), ("ᆱᄋ", "ᆯᄆ"), ("ᆲᄋ", "ᆯᄇ"), ("ᆳᄋ", "ᆯᄊ"), ("ᆴᄋ", "ᆯᄐ"), ("ᆵᄋ", "ᆯᄑ"), ("ᆹᄋ", "ᆸᄊ") };
        for (int i = 0; i < codas.Length; i++) inp = inp.Replace(codas[i] + "ᄋ", onsets[i]);          // link1
        foreach (var (from, to) in link2) inp = inp.Replace(from, to);                                 // link2
        if (_linkAcrossSpaces)
        {
            for (int i = 0; i < codas.Length; i++) inp = inp.Replace(codas[i] + " ᄋ", " " + onsets[i]);   // link3
            foreach (var (from, to) in link2) inp = inp.Replace(from.Insert(1, " "), to.Insert(1, " "));
        }
        foreach (var (from, to) in new[] { ("ᇂᄋ", "ᄋ"), ("ᆭᄋ", "ᄂ"), ("ᆶᄋ", "ᄅ") }) inp = inp.Replace(from, to);   // link4
        return inp;
    }
}

/// <summary>
/// MeloTTS's Korean front end (myshell-ai/MeloTTS <c>melo/text/korean.py</c>): CJK ideographs dropped, its two small
/// dictionaries applied, lower case; then kykim/bert-kor-base's word pieces, each word read by g2pkk and written as
/// jamo, one phone per jamo.
/// </summary>
internal sealed class MeloKoreanFrontEnd
{
    private static readonly Lazy<MeloKoreanFrontEnd> SharedInstance = new(() => new MeloKoreanFrontEnd(), isThreadSafe: true);
    private readonly WordPieceTokenizer _tokenizer = WordPieceTokenizer.BertKorBase();

    public static MeloKoreanFrontEnd Default => SharedInstance.Value;

    private static readonly string[] Punctuation = { "!", "?", "…", ",", ".", "'", "-" };
    private static readonly Regex Ideographs = new("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", RegexOptions.CultureInvariant);
    private static readonly (string From, string To)[] EtcDictionary = { ("1+1", "원플러스원"), ("2+1", "투플러스원") };
    private static readonly Dictionary<string, string> EnglishDictionary = new(StringComparer.Ordinal)
    {
        ["KOREA"] = "코리아", ["IDOL"] = "아이돌", ["IT"] = "아이티", ["IQ"] = "아이큐", ["UP"] = "업", ["DOWN"] = "다운",
        ["PC"] = "피씨", ["CCTV"] = "씨씨티비", ["SNS"] = "에스엔에스", ["AI"] = "에이아이", ["CEO"] = "씨이오", ["A"] = "에이",
        ["B"] = "비", ["C"] = "씨", ["D"] = "디", ["E"] = "이", ["F"] = "에프", ["G"] = "지", ["H"] = "에이치", ["I"] = "아이",
        ["J"] = "제이", ["K"] = "케이", ["L"] = "엘", ["M"] = "엠", ["N"] = "엔", ["O"] = "오", ["P"] = "피", ["Q"] = "큐",
        ["R"] = "알", ["S"] = "에스", ["T"] = "티", ["U"] = "유", ["V"] = "브이", ["W"] = "더블유", ["X"] = "엑스", ["Y"] = "와이",
        ["Z"] = "제트",
    };

    /// <summary><c>korean.normalize</c> (also <c>text_normalize</c>).</summary>
    public string Normalize(string text)
    {
        text = Ideographs.Replace(PythonText.Strip(text), "");
        if (EtcDictionary.Any(e => text.Contains(e.From)))
        {
            var pattern = new Regex(string.Join("|", EtcDictionary.Select(e => Regex.Escape(e.From))), RegexOptions.CultureInvariant);
            text = pattern.Replace(text, m => EtcDictionary.First(e => e.From == m.Value).To);
        }
        text = Regex.Replace(text, "([A-Za-z]+)", m => EnglishDictionary.TryGetValue(m.Value, out var hangul) ? hangul : m.Value);
        return PythonText.Lower(text);
    }

    /// <summary>The word pieces of kykim/bert-kor-base's tokenizer.</summary>
    public List<string> WordPieces(string text) => _tokenizer.Tokenize(text);

    /// <summary><c>korean_text_to_phonemes(text)</c>: normalized, read by g2pkk, written as jamo.</summary>
    public string TextToPhonemes(string text) => G2pKk.HangulToJamo(G2pKk.Default.Convert(Normalize(text)));

    /// <summary>
    /// <c>korean.g2p(norm_text)</c>: per word, its jamo (one phone each), "_" for an unknown piece and punctuation
    /// as itself, spread over the word's pieces, with the silence symbol at each end.
    /// </summary>
    public (List<string> Phones, List<int> Tones, List<int> PhonesPerPiece) G2P(string text)
    {
        var tokens = WordPieces(text);
        var groups = new List<List<string>>();
        foreach (var token in tokens)
        {
            if (!token.StartsWith("#", StringComparison.Ordinal)) groups.Add(new List<string> { token });
            else if (groups.Count == 0) throw new ArgumentException("The text starts with a word piece continuation.", nameof(text));
            else groups[groups.Count - 1].Add(token.Replace("#", ""));
        }
        var phones = new List<string>();
        var perPiece = new List<int>();
        foreach (var group in groups)
        {
            string word = string.Concat(group);
            if (word == "[UNK]")
            {
                phones.Add("_");
                perPiece.Add(1);
                continue;
            }
            if (Punctuation.Contains(word))
            {
                phones.Add(word);
                perPiece.Add(1);
                continue;
            }
            var jamo = ChineseText.Codepoints(TextToPhonemes(word));
            phones.AddRange(jamo);
            perPiece.AddRange(MeloEnglishFrontEnd.DistributePhones(jamo.Count, group.Count));
        }
        if (perPiece.Count != tokens.Count)
            throw new ArgumentException("The phones do not align with the word pieces.", nameof(text));
        phones.Insert(0, "_");
        phones.Add("_");
        perPiece.Insert(0, 1);
        perPiece.Add(1);
        return (phones, Enumerable.Repeat(0, phones.Count).ToList(), perPiece);
    }
}
