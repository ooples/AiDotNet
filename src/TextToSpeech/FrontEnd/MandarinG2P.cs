using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Converts Mandarin Chinese text to IPA phonemes, as VALL-E X's reference front end does (Plachtaa/VALL-E-X,
/// <c>utils/g2p/mandarin.py</c>, <c>chinese_to_ipa</c>).
/// </summary>
/// <remarks>
/// <para>
/// The pipeline is the reference's: Arabic numerals become Chinese numerals (cn2an's <c>an2cn</c>, "low" mode), jieba
/// segments the text into words (its prefix dictionary, maximum-probability route and HMM for unknown words), pypinyin
/// reads each word in BOPOMOFO (its phrase dictionary first, by forward maximum matching, then each character's first
/// reading), Latin letters are spelled in bopomofo, and a table maps bopomofo to IPA with tone contours
/// (→ ↑ ↓↑ ↓), followed by the reference's four rewrite rules (glides, apical vowels). The dictionaries are jieba 0.42.1's
/// and pypinyin 0.55.0's (MIT), exported by <c>tools/reference-data/mandarin_g2p_resources.py</c>.
/// </para>
/// <para>
/// <see cref="Phonemize"/> splits the IPA into the reference tokenizer's base symbols: one symbol per character, with
/// spaces as the word separator <c>_</c>.
/// </para>
/// <para><b>For Beginners:</b> Chinese characters do not spell their sounds, so this looks words up in a dictionary to
/// find each character's pronunciation (and tone) and writes it in the International Phonetic Alphabet.</para>
/// </remarks>
public sealed class MandarinG2P
{
    private static readonly Lazy<MandarinG2P> SharedInstance = new(() => new MandarinG2P(), isThreadSafe: true);

    private MandarinG2P()
    {
    }

    /// <summary>Every symbol <see cref="Phonemize"/> produces from Chinese, punctuation and Latin letters (characters it
    /// passes through unchanged, such as other symbols, are outside it).</summary>
    public static IReadOnlyList<string> Symbols => SymbolTable.Value;

    /// <summary>The shared instance (the dictionaries load once, on first use).</summary>
    public static MandarinG2P Default => SharedInstance.Value;

    /// <summary>The phonemes of <paramref name="text"/>: one symbol per IPA character, words separated by <c>_</c>.</summary>
    public IReadOnlyList<string> Phonemize(string text) =>
        ChineseText.Codepoints(ToIpa(text)).Select(c => c == " " ? "_" : c).ToList();

    /// <summary>The reference's <c>chinese_to_ipa</c> of <paramref name="text"/>.</summary>
    public string ToIpa(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        text = NumbersToChinese(text);
        text = ToBopomofo(text);
        text = LatinToBopomofo(text);
        foreach (var (pattern, replacement) in BopomofoToIpa) text = text.Replace(pattern, replacement);
        text = Regex.Replace(text, "i([aoe])", "j$1");
        text = Regex.Replace(text, "u([aoəe])", "w$1");
        text = Regex.Replace(text, "([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", "$1ɹ`$2").Replace("ɻ", "ɹ`");
        text = Regex.Replace(text, "([s][⁼ʰ]?)([→↓↑ ]+|$)", "$1ɹ$2");
        return text;
    }

    // ---------------------------------------------------------------- numerals (cn2an)

    private static readonly string[] Digits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
    private static readonly string[] Units = { "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千" };

    // number_to_chinese: each \d+(?:\.?\d+)? in turn, replacing its first occurrence.
    private static string NumbersToChinese(string text)
    {
        foreach (Match match in Regex.Matches(text, @"\d+(?:\.?\d+)?"))
        {
            int index = text.IndexOf(match.Value, StringComparison.Ordinal);
            text = text.Substring(0, index) + AnToCn(match.Value) + text.Substring(index + match.Value.Length);
        }
        return text;
    }

    /// <summary>cn2an's <c>an2cn</c> in its default "low" mode; <paramref name="joinThousands"/> false is the copy
    /// EmotiVoice vendors, which predates cn2an's rewrite of 万零X千 and 亿零X千 as 万X千 and 亿X千.</summary>
    internal static string AnToCn(string number, bool joinThousands = true)
    {
        foreach (char c in number)
            if (!(c is >= '0' and <= '9' || c == '.' || c == '-'))
                throw new ArgumentException($"'{number}' is outside cn2an's range (it reads ASCII digits).", nameof(number));
        string sign = number.StartsWith("-", StringComparison.Ordinal) ? "负" : "";
        if (sign.Length > 0) number = number.Substring(1);
        var parts = number.Split('.');
        if (parts.Length > 2) throw new ArgumentException($"Malformed number '{number}'.", nameof(number));
        string output = IntegerToCn(parts[0], joinThousands);
        if (parts.Length == 2)
        {
            string decimals = parts[1].Length > 16 ? parts[1].Substring(0, 16) : parts[1];
            output += decimals.Length > 0 ? "点" : "";
            foreach (char d in decimals) output += Digits[d - '0'];
        }
        return sign + output;
    }

    private static string IntegerToCn(string digits, bool joinThousands)
    {
        // str(int(...)): leading zeros dropped (an empty integer part is int("") in Python, an error).
        if (digits.Length == 0) throw new ArgumentException("cn2an needs an integer part.", nameof(digits));
        digits = digits.TrimStart('0');
        if (digits.Length == 0) digits = "0";
        int length = digits.Length;
        if (length > Units.Length) throw new ArgumentException($"cn2an reads at most {Units.Length} integer digits.", nameof(digits));
        var output = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            int d = digits[i] - '0';
            if (d != 0)
            {
                output.Append(Digits[d]).Append(Units[length - i - 1]);
            }
            else
            {
                if ((length - i - 1) % 4 == 0) output.Append(Digits[d]).Append(Units[length - i - 1]);
                if (i > 0 && (output.Length == 0 || output[output.Length - 1] != '零')) output.Append(Digits[d]);
            }
        }
        string result = output.ToString().Replace("零零", "零").Replace("零万", "万").Replace("零亿", "亿").Replace("亿万", "亿").Trim('零');
        if (joinThousands) result = Regex.Replace(result, "([万亿])零([一二三四五六七八九壹贰叁肆伍陆柒捌玖][千仟])", "$1$2");
        if (result.StartsWith("一十", StringComparison.Ordinal)) result = result.Substring(1);
        return result.Length == 0 ? "零" : result;
    }

    // ---------------------------------------------------------------- words and readings

    // chinese_to_bopomofo: jieba's words, each read by pypinyin; words without a CJK character pass through.
    private string ToBopomofo(string text)
    {
        text = text.Replace('、', '，').Replace('；', '，').Replace('：', '，');
        var output = new StringBuilder();
        foreach (var word in Jieba.Default.Cut(text))
        {
            if (!Regex.IsMatch(word, "[一-鿿]"))
            {
                output.Append(word);
                continue;
            }
            if (output.Length > 0) output.Append(' ');
            foreach (var reading in PypinyinReader.Bopomofo.Read(word))
                output.Append(Regex.Replace(reading, "([ㄅ-ㄩ])$", "$1ˉ"));
        }
        return output.ToString();
    }

    private static readonly (string Letter, string Bopomofo)[] LatinBopomofo =
    {
        ("a", "ㄟˉ"), ("b", "ㄅㄧˋ"), ("c", "ㄙㄧˉ"), ("d", "ㄉㄧˋ"), ("e", "ㄧˋ"), ("f", "ㄝˊㄈㄨˋ"), ("g", "ㄐㄧˋ"),
        ("h", "ㄝˇㄑㄩˋ"), ("i", "ㄞˋ"), ("j", "ㄐㄟˋ"), ("k", "ㄎㄟˋ"), ("l", "ㄝˊㄛˋ"), ("m", "ㄝˊㄇㄨˋ"), ("n", "ㄣˉ"),
        ("o", "ㄡˉ"), ("p", "ㄆㄧˉ"), ("q", "ㄎㄧㄡˉ"), ("r", "ㄚˋ"), ("s", "ㄝˊㄙˋ"), ("t", "ㄊㄧˋ"), ("u", "ㄧㄡˉ"),
        ("v", "ㄨㄧˉ"), ("w", "ㄉㄚˋㄅㄨˋㄌㄧㄡˋ"), ("x", "ㄝˉㄎㄨˋㄙˋ"), ("y", "ㄨㄞˋ"), ("z", "ㄗㄟˋ"),
    };

    private static string LatinToBopomofo(string text)
    {
        foreach (var (letter, bopomofo) in LatinBopomofo)
            text = Regex.Replace(text, letter, bopomofo, RegexOptions.IgnoreCase);
        return text;
    }

    private static readonly (string Pattern, string Replacement)[] BopomofoToIpa =
    {
        ("ㄅㄛ", "p⁼wo"), ("ㄆㄛ", "pʰwo"), ("ㄇㄛ", "mwo"), ("ㄈㄛ", "fwo"), ("ㄅ", "p⁼"), ("ㄆ", "pʰ"), ("ㄇ", "m"),
        ("ㄈ", "f"), ("ㄉ", "t⁼"), ("ㄊ", "tʰ"), ("ㄋ", "n"), ("ㄌ", "l"), ("ㄍ", "k⁼"), ("ㄎ", "kʰ"), ("ㄏ", "x"),
        ("ㄐ", "tʃ⁼"), ("ㄑ", "tʃʰ"), ("ㄒ", "ʃ"), ("ㄓ", "ts`⁼"), ("ㄔ", "ts`ʰ"), ("ㄕ", "s`"), ("ㄖ", "ɹ`"),
        ("ㄗ", "ts⁼"), ("ㄘ", "tsʰ"), ("ㄙ", "s"), ("ㄚ", "a"), ("ㄛ", "o"), ("ㄜ", "ə"), ("ㄝ", "ɛ"), ("ㄞ", "aɪ"),
        ("ㄟ", "eɪ"), ("ㄠ", "ɑʊ"), ("ㄡ", "oʊ"), ("ㄧㄢ", "jɛn"), ("ㄩㄢ", "ɥæn"), ("ㄢ", "an"), ("ㄧㄣ", "in"),
        ("ㄩㄣ", "ɥn"), ("ㄣ", "ən"), ("ㄤ", "ɑŋ"), ("ㄧㄥ", "iŋ"), ("ㄨㄥ", "ʊŋ"), ("ㄩㄥ", "jʊŋ"), ("ㄥ", "əŋ"),
        ("ㄦ", "əɻ"), ("ㄧ", "i"), ("ㄨ", "u"), ("ㄩ", "ɥ"), ("ˉ", "→"), ("ˊ", "↑"), ("ˇ", "↓↑"), ("ˋ", "↓"),
        ("˙", ""), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("—", "-"),
    };

    // Declared after the table it reads: static fields initialize in textual order.
    private static readonly Lazy<IReadOnlyList<string>> SymbolTable = new(() =>
    {
        var symbols = new SortedSet<string>(StringComparer.Ordinal) { "_", "j", "w", "ɹ", "`" };
        foreach (var (_, replacement) in BopomofoToIpa)
            foreach (var c in ChineseText.Codepoints(replacement)) symbols.Add(c);
        return symbols.ToList();
    }, isThreadSafe: true);

}
